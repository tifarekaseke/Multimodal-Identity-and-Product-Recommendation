from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "images" / "raw"
STD_DIR = ROOT / "images" / "standardized"
AUG_DIR = ROOT / "images" / "augmented"
RESULTS_DIR = ROOT / "results"
OUT_CSV = ROOT / "data" / "image_features.csv"
SIZE = (224, 224)
EXPRESSIONS = ["neutral", "smiling", "surprised"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_name(path):
    member, expression = path.stem.lower().rsplit("_", 1)
    return member, expression


def standardize_images():
    STD_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(RAW_DIR.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        member, expression = parse_name(path)
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, SIZE)
        cv2.imwrite(str(STD_DIR / f"{member}_{expression}.jpg"), img)
        items.append({"member": member, "expression": expression, "image": img})
    return items


def rotate(img, angle=15):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))


def flip(img):
    return cv2.flip(img, 1)


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


AUGMENTATIONS = {"rotate": rotate, "flip": flip, "grayscale": grayscale}


def augment_images(items):
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        source = f"{item['member']}_{item['expression']}"
        rows.append({**item, "augmentation": "original", "source_image": source})
        for name, func in AUGMENTATIONS.items():
            aug = func(item["image"])
            cv2.imwrite(str(AUG_DIR / f"{source}_{name}.jpg"), aug)
            rows.append({**item, "image": aug, "augmentation": name, "source_image": source})
    return rows


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()
    features = {f"hist_{i}": value for i, value in enumerate(hist)}
    features["mean_intensity"] = float(gray.mean())
    features["std_intensity"] = float(gray.std())
    return features


def build_feature_table(rows):
    records = []
    for row in rows:
        record = {
            "source_image": row["source_image"],
            "member": row["member"],
            "expression": row["expression"],
            "augmentation": row["augmentation"],
        }
        record.update(extract_features(row["image"]))
        records.append(record)
    df = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def show_samples(items):
    members = sorted({item["member"] for item in items})
    lookup = {(i["member"], i["expression"]): i["image"] for i in items}
    fig, axes = plt.subplots(len(members), 3, figsize=(9, 3 * len(members)), squeeze=False)
    for r, member in enumerate(members):
        for c, expression in enumerate(EXPRESSIONS):
            ax = axes[r][c]
            ax.axis("off")
            img = lookup.get((member, expression))
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{member} - {expression}")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "sample_faces.png", dpi=150)
    return fig


def show_augmentations(rows):
    first = rows[0]["source_image"]
    subset = [r for r in rows if r["source_image"] == first]
    fig, axes = plt.subplots(1, len(subset), figsize=(4 * len(subset), 4))
    for ax, row in zip(axes, subset):
        ax.imshow(cv2.cvtColor(row["image"], cv2.COLOR_BGR2RGB))
        ax.set_title(row["augmentation"])
        ax.axis("off")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "augmentation_example.png", dpi=150)
    return fig


def main():
    items = standardize_images()
    rows = augment_images(items)
    df = build_feature_table(rows)
    plt.close(show_samples(items))
    plt.close(show_augmentations(rows))
    print(f"Images found: {len(items)}")
    print(f"Rows saved to {OUT_CSV.name}: {len(df)}")
    print(df[["source_image", "augmentation"]].head(8))


if __name__ == "__main__":
    main()