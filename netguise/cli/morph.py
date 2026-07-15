from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from scapy.packet import Raw
from scapy.utils import rdpcap, wrpcap

from netguise.checkpoints import load_checkpoint
from netguise.data import (
    extract_dominant_flow,
    load_processed,
    register_raw_ipv4_linktype,
)
from netguise.pcap import apply_timing_plan, write_plan, write_raw_ipv4_pcap
from netguise.reference import ReferencePool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a NetGuise morphing plan")
    parser.add_argument("--input", required=True, help="source PCAP")
    parser.add_argument("--plan", default="results/morphing-plan.json")
    parser.add_argument("--output-pcap", default=None)
    parser.add_argument("--model", "--model_path", default="models/morpher/netguise.pt")
    parser.add_argument("--reference-data", default="data/processed_dataset")
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--mode", choices=("full", "online"), default="full")
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def _class_index(value: str, classes: np.ndarray) -> int:
    if value in classes.astype(str):
        return int(np.flatnonzero(classes.astype(str) == value)[0])
    index = int(value)
    if index < 0 or index >= len(classes):
        raise ValueError(f"target class index {index} is out of range")
    return index


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint = load_checkpoint(args.model, device=device)
    if checkpoint["model_name"] != "netguise":
        raise SystemExit("--model must point to a NetGuise morpher checkpoint")
    register_raw_ipv4_linktype()
    packets = rdpcap(args.input)
    extracted = extract_dominant_flow(packets, max_packets=model.max_packets)
    if extracted is None:
        raise SystemExit("the input contains no TCP/UDP flow")
    pool_x, pool_y, classes = load_processed(args.reference_data)
    pool_x = pool_x[:, : model.max_packets]
    target_class = _class_index(args.target_class, classes)
    observed = 1 if args.mode == "online" else None
    target, reference_index, score = ReferencePool(pool_x, pool_y).select(
        extracted.features,
        target_class,
        candidate_count=args.candidate_count,
        seed=args.seed,
        observed_packets=observed,
    )
    source_tensor = torch.as_tensor(
        extracted.features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        output = (
            model.forward_causal(source_tensor, target_tensor)
            if args.mode == "online"
            else model(source_tensor, target_tensor)
        )
    actions = output.actions[0].cpu().numpy()
    morphed = output.morphed[0].cpu().numpy()
    write_plan(
        args.plan,
        extracted.features,
        target,
        morphed,
        actions,
        {
            "interface": args.mode,
            "target_class": int(target_class),
            "target_name": str(classes[target_class]),
            "reference_index": int(reference_index),
            "reference_score": score,
            "runtime_classifier_queries": 0,
            "padding_materialized": False,
        },
    )
    if args.output_pcap:
        shaped = apply_timing_plan(packets, extracted.packet_indices, actions)
        Path(args.output_pcap).parent.mkdir(parents=True, exist_ok=True)
        if shaped and isinstance(shaped[0], Raw):
            write_raw_ipv4_pcap(args.output_pcap, shaped)
        else:
            wrpcap(str(args.output_pcap), shaped)
        logging.info(
            "wrote timing-shaped PCAP; padding remains in the plan for tunnel-layer application"
        )
    logging.info("wrote %s", args.plan)


if __name__ == "__main__":
    main()
