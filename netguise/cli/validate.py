from __future__ import annotations

import argparse
import json

import numpy as np

from netguise.pcap import load_plan
from netguise.validation import validate_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a NetGuise morphing plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mtu", type=float, default=1500.0)
    parser.add_argument("--bandwidth-ratio", type=float, default=0.30)
    parser.add_argument("--latency-ratio", type=float, default=0.50)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    report = validate_features(
        np.asarray(plan["source"], dtype=np.float32),
        np.asarray(plan["morphed"], dtype=np.float32),
        mtu=args.mtu,
        bandwidth_ratio=args.bandwidth_ratio,
        latency_ratio=args.latency_ratio,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
