import argparse
import numpy as np


def Main():
    parser = argparse.ArgumentParser(description="concat nonflip .npy corpora along case axis")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parts = [np.load(p, mmap_mode="r") for p in args.inputs]
    for p, a in zip(args.inputs, parts):
        print(f"[concat] {p}: {a.shape} {a.dtype}")
        if a.shape[1:] != parts[0].shape[1:]:
            raise ValueError(f"shape mismatch: {p} {a.shape} vs {parts[0].shape}")
        if not np.allclose(a[0, :, 0], parts[0][0, :, 0]):
            raise ValueError(f"t_grid mismatch: {p}")
    out = np.concatenate([np.asarray(a, dtype=np.float32) for a in parts], axis=0)
    np.save(args.out, out)
    print(f"[concat] saved {out.shape} -> {args.out}")


if __name__ == "__main__":
    Main()
