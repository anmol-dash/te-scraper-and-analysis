import sys

from yourtool.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]) or 0)
