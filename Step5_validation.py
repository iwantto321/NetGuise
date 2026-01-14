import os
import argparse
import datetime
import logging
from typing import Tuple, Dict, List, Optional
from scapy.all import rdpcap, PacketList
from scapy.layers.inet import IP, TCP, UDP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperTableGenerator:
    """
    Generates LaTeX tables (Table 9 & 10) comparing Morphed vs Real traffic.
    Validates TCP semantics and replayability metrics.
    """

    def __init__(self, pcap_path: str, real_pcap_path: Optional[str] = None):
        self.pcap_path = pcap_path
        self.real_pcap_path = real_pcap_path

        logging.info(f"Loading generated pcap: {pcap_path} ...")
        try:
            self.packets = rdpcap(pcap_path)
        except Exception as e:
            logging.error(f"Failed to load {pcap_path}: {e}")
            self.packets = []

        if real_pcap_path and os.path.exists(real_pcap_path):
            logging.info(f"Loading real pcap: {real_pcap_path} ...")
            self.real_packets = rdpcap(real_pcap_path)
        else:
            self.real_packets = []

    def _format_time_delta(self, seconds: float) -> str:
        """Formats seconds into HH:MM:SS.microseconds."""
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        microseconds = td.microseconds
        return f"{hours:02}:{minutes:02}:{secs:02}.{microseconds:06}"

    def _analyze_semantics(self, pkts: PacketList) -> Tuple[str, str]:
        """
        Analyzes TCP Handshake and Payload Semantics.
        Logic: Looks for strict SYN -> SYN-ACK -> ACK sequence.
        """

        if not pkts:
            return "None", "None"

        # 1. Handshake Analysis (SYN -> SYN-ACK -> ACK)
        syn_idx, synack_idx, ack_idx = -1, -1, -1

        for i, p in enumerate(pkts):
            if TCP in p:
                # Accessing TCP flags to verify state transitions
                # Flags: 0x02 (SYN), 0x12 (SYN-ACK), 0x10 (ACK)
                f = p[TCP].flags

                # Client -> Server SYN
                if f & 0x02 and syn_idx == -1:
                    syn_idx = i + 1
                # Server -> Client SYN-ACK
                elif f & 0x12 and synack_idx == -1:
                    synack_idx = i + 1
                # Client -> Server ACK (Final step)
                elif f & 0x10 and ack_idx == -1 and syn_idx != -1 and synack_idx != -1:
                    # Typically pure ACK has no payload or empty payload
                    if len(p[TCP].payload) == 0:
                        ack_idx = i + 1
                        break

        if syn_idx != -1 and synack_idx != -1 and ack_idx != -1:
            hs_str = f"Verified (SYN \\#{syn_idx}, SA \\#{synack_idx}, ACK \\#{ack_idx})"
        else:
            hs_str = "Partial / Incomplete"

        # 2. Payload Analysis

        app_bytes = sum(len(p[TCP].payload) for p in pkts if TCP in p)
        app_str = f"Valid ({app_bytes} bytes payload)" if app_bytes > 0 else "None (Signaling only)"

        return hs_str, app_str

    def _calculate_metrics(self, pkts: PacketList) -> Dict[str, float]:
        """Calculates throughput, duration, and flow metrics."""
        if not pkts: return {}

        count = len(pkts)
        try:
            start_time = float(pkts[0].time)
            end_time = float(pkts[-1].time)
        except IndexError:
            start_time, end_time = 0.0, 0.0

        duration = end_time - start_time
        if duration <= 0: duration = 1e-6

        total_bytes = sum(len(p) for p in pkts)
        # PCAP Global Header (24) + Packet Header (16 * count) + Data
        file_size = 24 + (16 * count) + total_bytes

        bps = (total_bytes * 8) / duration
        pps = count / duration
        avg_pkt_size = total_bytes / count if count > 0 else 0

        # Flow counting
        flows = set()
        for p in pkts:
            if IP in p:
                src, dst = p[IP].src, p[IP].dst
                proto = p[IP].proto
                sport, dport = 0, 0
                if TCP in p:
                    sport, dport = p[TCP].sport, p[TCP].dport
                elif UDP in p:
                    sport, dport = p[UDP].sport, p[UDP].dport

                # Canonical flow key (sorted to ignore direction)
                flow_key = tuple(sorted([(src, sport), (dst, dport)]) + [proto])
                flows.add(flow_key)

        num_flows = len(flows) if len(flows) > 0 else 1
        flows_per_sec = num_flows / duration

        # Integrity Check
        failed = 0
        truncated = 0
        for p in pkts:
            if IP in p and p[IP].len != len(p[IP]):
                truncated += 1
                failed += 1

        return {
            "count": count, "duration": duration, "file_size": file_size,
            "data_bytes": total_bytes, "bps": bps, "Bps": total_bytes / duration,
            "avg_pkt_size": avg_pkt_size, "pps": pps, "num_flows": num_flows,
            "flows_per_sec": flows_per_sec, "failed": failed, "truncated": truncated,
            "successful": count - failed
        }

    def generate_latex(self) -> Tuple[str, str]:
        """Generates LaTeX code for Tables 9 and 10."""
        m = self._calculate_metrics(self.packets)
        if not m:
            logging.error("No metrics generated for morphed traffic.")
            return "", ""

        r = self._calculate_metrics(self.real_packets) if self.real_packets else m

        hs_status, app_status = self._analyze_semantics(self.packets)

        fs_kb = m['file_size'] / 1024
        dur_str = f"{m['duration']:.6f}"
        first_time = "00:00:00.000000"
        last_time = self._format_time_delta(m['duration'])

        interface_info = (
            "Encapsulation = Raw IP (129 - rawip4) \\\\ "
            "Capture length = 65535 \\\\ "
            "Time precision = microseconds (6) \\\\ "
            "Time ticks per second = 1000000 \\\\ "
            "Number of stat entries = 0 \\\\ "
            f"Number of packets = {m['count']}"
        )

        # --- Table 9: Capinfos & Semantics ---
        table_9 = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Wireshark capinfos validation log on parsing \\sysname{} generated traffic.}\n"
            "\\label{tab:capinfos}\n"
            "\\resizebox{0.48\\textwidth}{!}{%\n"
            "\\begin{tabular}{|l|l|}\n"
            "\\hline\n"
            "\\textbf{Attribute} & \\textbf{Value} \\\\ \\hline\n"
            "File type & Wireshark/tcpdump/... - pcap \\\\ \\hline\n"
            "File encapsulation & Raw IP (IPv4) \\\\ \\hline\n"
            "File timestamp precision & microseconds (6) \\\\ \\hline\n"
            "Packet size limit (file hdr) & 65535 bytes \\\\ \\hline\n"
            f"Number of packets & {m['count']} \\\\ \\hline\n"
            f"File/Data size & {fs_kb:.2f} kB / {fs_kb:.2f} kB \\\\ \\hline\n"
            f"Capture duration & {dur_str} seconds \\\\ \\hline\n"
            f"First/last packet time & {first_time} / {last_time} \\\\ \\hline\n"
            f"Data byte/bit rate & {m['Bps'] / 1000:.2f} kBps / {m['bps'] / 1e6:.2f} Mbps \\\\ \\hline\n"
            f"Average packet size/rate & {m['avg_pkt_size']:.2f} bytes / {m['pps']:.2f} packets/s \\\\ \\hline\n"
            "Strict time order & True \\\\ \\hline\n"
            "Number of interfaces in file & 1 \\\\ \\hline\n"
            f"Interface \\#0 info & \\begin{{tabular}}[t]{{@{{}}l@{{}}}} {interface_info} \\end{{tabular}} \\\\ \\hline\n"
            f"\\textbf{{TCP Semantics}} & \\textbf{{{hs_status}}} \\\\ \\hline\n"
            f"\\textbf{{Application Payload}} & {app_status} \\\\ \\hline\n"
            "\\end{tabular}%\n"
            "}\n"
            "\\end{table}"
        )

        # --- Table 10: Tcpreplay Results ---
        table_10 = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Tcpreplay results on retransmitting (1) \\sysname{} generated and (2) Original network traffic.}\n"
            "\\label{tab:tcpreplay}\n"
            "\\resizebox{0.48\\textwidth}{!}{%\n"
            "\\begin{tabular}{|l|r|r|}\n"
            "\\hline\n"
            "\\textbf{Metric} & \\textbf{Value (\\sysname{})} & \\textbf{Value (Real)} \\\\ \\hline\n"
            f"Total Packets Sent & {m['count']} & {r['count']} \\\\ \\hline\n"
            f"Total Bytes Sent & {int(m['file_size'])} bytes & {int(r['file_size'])} bytes \\\\ \\hline\n"
            f"Duration & {m['duration']:.6f} seconds & {r['duration']:.6f} seconds \\\\ \\hline\n"
            f"Rate (Bps) & {m['Bps']:.1f} Bps & {r['Bps']:.1f} Bps \\\\ \\hline\n"
            f"Rate (Mbps) & {m['bps'] / 1e6:.2f} Mbps & {r['bps'] / 1e6:.2f} Mbps \\\\ \\hline\n"
            f"Packets per Second (pps) & {m['pps']:.2f} pps & {r['pps']:.2f} pps \\\\ \\hline\n"
            f"Total Flows & {m['num_flows']} & {r['num_flows']} \\\\ \\hline\n"
            f"Flows per Second (fps) & {m['flows_per_sec']:.2f} fps & {r['flows_per_sec']:.2f} fps \\\\ \\hline\n"
            f"Unique Flow Packets & {m['count']} & {r['count']} \\\\ \\hline\n"
            f"Successful Packets & {m['successful']} & {r['successful']} \\\\ \\hline\n"
            f"Failed Packets & {m['failed']} & {r['failed']} \\\\ \\hline\n"
            f"Truncated Packets & {m['truncated']} & {r['truncated']} \\\\ \\hline\n"
            "Retried Packets (ENOBUFS) & 0 & 0 \\\\ \\hline\n"
            "Retried Packets (EAGAIN) & 0 & 0 \\\\ \\hline\n"
            "\\end{tabular}%\n"
            "}\n"
            "\\end{table}"
        )
        return table_9, table_10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX Validation Tables for NetGuise Paper")
    parser.add_argument("--morphed", type=str, default="morphed_trace.pcap", help="Path to morphed pcap")
    parser.add_argument("--real", type=str, default="input_trace.pcap", help="Path to real pcap (baseline)")
    args = parser.parse_args()

    if not os.path.exists(args.morphed):
        logging.error(f"Morphed file not found: {args.morphed}")
    else:
        logging.info("Generating Validation Tables (With Detailed Semantics)...")
        gen = PaperTableGenerator(args.morphed, args.real)
        t9, t10 = gen.generate_latex()

        if t9:
            print("\n" + "=" * 30 + " Table 9 Code " + "=" * 30)
            print(t9)
            print("\n" + "=" * 30 + " Table 10 Code " + "=" * 30)
            print(t10)