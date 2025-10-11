import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()

    data = torch.load(args.checkpoint, map_location="cpu")
    best = data.get("best_acc1", None)
    if best is None:
        print("best_acc1 not found")
    else:
        value = best.item() if hasattr(best, "item") else best
        print(f"best_acc1: {value}")

if __name__ == "__main__":
    main()