#!/usr/bin/env python3
"""CLI wrapper for identification. Called by the Express API.

Usage:
    python identify_cli.py <image_path> [--top-k 5] [--herd-id 3]

Outputs JSON array of predictions to stdout.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from identifier import identify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--herd-id", type=int, default=None)
    args = parser.parse_args()

    result = identify(
        image_path=args.image_path,
        herd_id=args.herd_id,
        top_k=args.top_k,
    )

    json.dump(result["predictions"], sys.stdout)


if __name__ == "__main__":
    main()
