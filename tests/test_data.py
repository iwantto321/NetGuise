from pathlib import Path

from scapy.layers.inet import IP, TCP
from scapy.packet import Raw
from scapy.utils import wrpcap

from netguise.data import extract_pcap


def test_bidirectional_flow_keeps_endpoint_pairs(tmp_path: Path):
    packets = [
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=50000, dport=443) / Raw(b"a"),
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=443, dport=50000) / Raw(b"bb"),
    ]
    packets[0].time = 1.0
    packets[1].time = 1.2
    path = tmp_path / "sample.pcap"
    wrpcap(str(path), packets)

    extracted = extract_pcap(path, max_packets=4)
    assert extracted is not None
    assert len(extracted.packet_indices) == 2
    assert extracted.features[0, 1] == 1.0
    assert extracted.features[1, 1] == -1.0
    assert abs(float(extracted.features[1, 2]) - 0.2) < 1e-6
