"""Allow running as `python -m arxiv_parser`."""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
