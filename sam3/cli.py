"""
SAM3 command-line interface (stub -- will be implemented in Step 16).
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sam3`` CLI."""
    parser = argparse.ArgumentParser(
        prog="sam3",
        description="SAM3 -- Segment Anything Model 3 CLI",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )

    # Placeholder subcommand group
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("segment", help="Run segmentation inference")

    args = parser.parse_args(argv)

    if args.version:
        from sam3 import __version__

        print(f"sam3 {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    # Will be filled in Step 16
    print(f"Command '{args.command}' is not yet implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
