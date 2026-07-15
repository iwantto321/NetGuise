from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from netguise.data import discover_samples, extract_pcap, save_processed


def _extract(sample: tuple[Path, int, str], max_packets: int):
    path, label, class_name = sample
    result = extract_pcap(path, max_packets=max_packets)
    return None if result is None else (result.features, label, class_name, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract early-flow packet features")
    parser.add_argument("--data-root", "--data_root", required=True)
    parser.add_argument(
        "--output-dir", "--save_dir", default="data/processed_dataset"
    )
    parser.add_argument("--max-packets", "--target_n", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    samples = discover_samples(args.data_root)
    if not samples:
        raise SystemExit(f"no class-organized PCAP files found under {args.data_root}")
    classes = sorted({class_name for _, _, class_name in samples})
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    _extract,
                    samples,
                    [args.max_packets] * len(samples),
                )
            )
    else:
        results = [_extract(sample, args.max_packets) for sample in samples]
    valid = [result for result in results if result is not None]
    if not valid:
        raise SystemExit("no valid TCP/UDP flows were extracted")
    save_processed(
        args.output_dir,
        (result[0] for result in valid),
        (result[1] for result in valid),
        classes,
    )
    logging.info("saved %d flows to %s", len(valid), args.output_dir)


if __name__ == "__main__":
    main()
