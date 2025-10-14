import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser(description="Trains the model.")
    #ap.add_argument("--scroll", type=int, required=True, help="Scroll identifier.")
    args = ap.parse_args()

if __name__ == "__main__":
    main()
