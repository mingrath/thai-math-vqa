"""Data loading + robust image-path resolution for Thai Math VQA."""
import os
import pandas as pd

# The CSV `image_path` column says e.g. "images/0.jpg", but after unzipping the
# Kaggle archive into data/ the files actually live at data/images/images/0.jpg.
# We try a few candidate roots so the pipeline works regardless of nesting.
def resolve_image_path(data_dir: str, image_path: str) -> str:
    candidates = [
        os.path.join(data_dir, "images", image_path),  # data/images/images/0.jpg  (actual)
        os.path.join(data_dir, image_path),            # data/images/0.jpg
        os.path.join(data_dir, "images", os.path.basename(image_path)),
        os.path.join(data_dir, os.path.basename(image_path)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # default to the most-likely path even if missing (error surfaces later, clearly)
    return candidates[0]


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    """split in {'train','test'}. Returns df with columns: id, image_path, abs_path[, answer]."""
    csv_path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(csv_path)
    df["abs_path"] = df["image_path"].apply(lambda p: resolve_image_path(data_dir, p))
    missing = (~df["abs_path"].apply(os.path.exists)).sum()
    if missing:
        raise FileNotFoundError(
            f"{missing}/{len(df)} images for split '{split}' not found under {data_dir!r}. "
            f"First expected: {df['abs_path'].iloc[0]!r}"
        )
    return df


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data"
    for sp in ("train", "test"):
        df = load_split(d, sp)
        print(f"{sp}: {len(df)} rows | cols={list(df.columns)}")
        print("  e.g.", df.iloc[0].to_dict())
