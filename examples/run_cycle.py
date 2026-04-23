"""
Example: run a full prediction cycle from the command line.

  python examples/run_cycle.py --lat 40.42 --lon -3.70 --name "Madrid"

This script demonstrates the *inference* path with the pre-trained
ensemble. It does NOT train the short-term / mid-term models; those
require either (a) a prior training run that fits them on historical
data (see scripts/train.py — not included here) or (b) letting the
orchestrator fall back to raw-consensus-only, which is exactly what
happens below when `short_model`/`mid_model` are None.

Even in that fallback, the system still produces calibrated intervals
(built from per-source agreement), NLP-enriched explanations, and
alerts — which is what demonstrates the value of the multi-source
ingestion and ensemble design independently of model training.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging

from src.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    orch = Orchestrator(args.config)
    result = asyncio.run(orch.run_cycle(args.lat, args.lon, args.name))

    print("\n────── FORECAST ──────")
    # Print first 6 per variable for brevity
    from collections import defaultdict
    by_var = defaultdict(list)
    for p in result.predictions:
        by_var[p.variable].append(p)

    for var, preds in by_var.items():
        print(f"\n{var}:")
        for p in preds[:6]:
            print(f"  +{p.lead_hours:>4.0f}h  "
                  f"{p.point:>6.1f}  [{p.lower:>6.1f}, {p.upper:>6.1f}]  "
                  f"conf={p.confidence_pct:>4.0f}%  ({p.horizon})")

    print("\n────── EXPLANATIONS (top 3) ──────")
    for e in result.explanations[:3]:
        print(f"• {e}")

    print("\n────── ALERTS ──────")
    if not result.alerts:
        print("(none)")
    for a in result.alerts:
        print(f"[{a['severity'].upper()}] {a['code']}: {a['message']}")

    print("\n────── ANOMALY ──────")
    print(json.dumps(result.anomaly, indent=2))

    print("\n────── GEO ──────")
    print(json.dumps(result.geo, indent=2))


if __name__ == "__main__":
    main()
