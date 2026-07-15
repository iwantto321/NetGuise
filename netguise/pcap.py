from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def apply_timing_plan(
    packets: Sequence[object],
    packet_indices: Sequence[int],
    actions: np.ndarray,
):
    """Apply only delays to a trace while preserving global packet order.

    Padding is intentionally kept in the plan unless the caller owns a tunnel
    encapsulation layer. Appending bytes to an arbitrary TCP payload would not
    preserve application semantics.
    """

    output = [copy.deepcopy(packet) for packet in packets]
    delay_by_index = {
        int(packet_index): float(actions[position, 1])
        for position, packet_index in enumerate(packet_indices)
        if position < len(actions)
    }
    accumulated_delay = 0.0
    previous_time = None
    for index, packet in enumerate(output):
        accumulated_delay += max(0.0, delay_by_index.get(index, 0.0))
        timestamp = float(packet.time) + accumulated_delay
        if previous_time is not None:
            timestamp = max(timestamp, previous_time)
        packet.time = timestamp
        previous_time = timestamp
    return output


def write_raw_ipv4_pcap(path: str | Path, packets: Sequence[object]) -> None:
    """Write Raw IPv4 packets with timestamps and DLT 228."""

    from scapy.utils import RawPcapWriter

    with RawPcapWriter(str(path), linktype=228, sync=True) as writer:
        writer.write_header(None)
        for packet in packets:
            timestamp = float(packet.time)
            seconds = int(timestamp)
            microseconds = int(round((timestamp - seconds) * 1_000_000))
            if microseconds == 1_000_000:
                seconds += 1
                microseconds = 0
            writer.write_packet(
                bytes(packet), sec=seconds, usec=microseconds, wirelen=len(packet)
            )


def write_plan(
    path: str | Path,
    source: np.ndarray,
    target: np.ndarray,
    morphed: np.ndarray,
    actions: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "format_version": 1,
        "metadata": metadata,
        "source": source.tolist(),
        "target_reference": target.tolist(),
        "morphed": morphed.tolist(),
        "actions": actions.tolist(),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_plan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
