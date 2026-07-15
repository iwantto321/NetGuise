from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


def register_raw_ipv4_linktype() -> None:
    from scapy.config import conf
    from scapy.layers.inet import IP

    conf.l2types.register(228, IP)


@dataclass(frozen=True)
class FlowFeatures:
    features: np.ndarray
    packet_indices: tuple[int, ...]
    flow_key: tuple[object, ...]


def _packet_layers(packet):
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.packet import Raw

    if IP in packet:
        network = packet[IP]
        decoded = packet
    elif IPv6 in packet:
        network = packet[IPv6]
        decoded = packet
    elif Raw in packet:
        payload = bytes(packet[Raw])
        if not payload:
            return None
        version = payload[0] >> 4
        if version == 4:
            network = IP(payload)
        elif version == 6:
            network = IPv6(payload)
        else:
            return None
        decoded = network
    else:
        return None
    if TCP in decoded:
        transport = decoded[TCP]
        protocol = "tcp"
    elif UDP in decoded:
        transport = decoded[UDP]
        protocol = "udp"
    else:
        return None
    return network, transport, protocol


def _canonical_flow_key(packet) -> Optional[tuple[object, ...]]:
    layers = _packet_layers(packet)
    if layers is None:
        return None
    network, transport, protocol = layers
    endpoint_a = (str(network.src), int(transport.sport))
    endpoint_b = (str(network.dst), int(transport.dport))
    low, high = sorted((endpoint_a, endpoint_b))
    return low + high + (protocol,)


def extract_dominant_flow(
    packets: Sequence[object], max_packets: int = 30
) -> Optional[FlowFeatures]:
    """Extract size, direction, and IAT from the largest bidirectional flow."""

    flows: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, packet in enumerate(packets):
        key = _canonical_flow_key(packet)
        if key is not None:
            flows[key].append(index)
    if not flows:
        return None

    flow_key = max(flows, key=lambda key: len(flows[key]))
    indices = sorted(flows[flow_key], key=lambda idx: float(packets[idx].time))
    chosen = indices[:max_packets]
    first_layers = _packet_layers(packets[chosen[0]])
    assert first_layers is not None
    client_address = str(first_layers[0].src)

    features = np.zeros((max_packets, 3), dtype=np.float32)
    previous_time = float(packets[chosen[0]].time)
    for output_index, packet_index in enumerate(chosen):
        packet = packets[packet_index]
        layers = _packet_layers(packet)
        assert layers is not None
        network, _, _ = layers
        if getattr(network, "version", 4) == 4:
            size = int(network.len) if network.len is not None else len(network)
        else:
            size = int(network.plen or 0) + 40
        timestamp = float(packet.time)
        iat = max(0.0, timestamp - previous_time) if output_index else 0.0
        previous_time = timestamp
        direction = 1.0 if str(network.src) == client_address else -1.0
        features[output_index] = (size, direction, iat)
    return FlowFeatures(features, tuple(chosen), flow_key)


def extract_pcap(path: str | Path, max_packets: int = 30) -> Optional[FlowFeatures]:
    from scapy.utils import rdpcap

    register_raw_ipv4_linktype()
    try:
        packets = rdpcap(str(path))
    except (OSError, ValueError) as error:
        LOGGER.warning("could not read %s: %s", path, error)
        return None
    return extract_dominant_flow(packets, max_packets=max_packets)


def discover_samples(data_root: str | Path) -> list[tuple[Path, int, str]]:
    root = Path(data_root)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    samples: list[tuple[Path, int, str]] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        paths = sorted(class_dir.glob("*.pcap")) + sorted(class_dir.glob("*.pcapng"))
        samples.extend((path, label, class_name) for path in paths)
    return samples


def load_processed(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_dir)
    x = np.load(root / "X_data.npy")
    y = np.load(root / "y_label.npy")
    classes = np.load(root / "classes.npy", allow_pickle=False)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"expected X_data.npy with shape (N, T, 3), got {x.shape}")
    if len(x) != len(y):
        raise ValueError("X_data.npy and y_label.npy have different sample counts")
    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False), classes


def feature_statistics(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = x[..., 0] > 0
    rows = x[valid]
    if not len(rows):
        raise ValueError("dataset has no valid packets")
    mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = rows.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def save_processed(
    save_dir: str | Path,
    features: Iterable[np.ndarray],
    labels: Iterable[int],
    classes: Iterable[str],
) -> None:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    x = np.asarray(list(features), dtype=np.float32)
    y = np.asarray(list(labels), dtype=np.int64)
    np.save(output / "X_data.npy", x)
    np.save(output / "y_label.npy", y)
    np.save(output / "classes.npy", np.asarray(list(classes)))
