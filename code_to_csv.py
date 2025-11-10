import os
import pandas as pd

# CHANGE THIS PATH TO YOUR DATASET FOLDER
root = r"C:\Users\Arsh\Downloads\chest_xray\chest_xray"

splits = ["train", "val", "test"]
labels_map = {"NORMAL": 0, "PNEUMONIA": 1}

def collect(split):
    rows = []
    for label_name, label in labels_map.items():
        folder = os.path.join(root, split, label_name)
        if not os.path.exists(folder):
            print("Missing folder:", folder)
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(folder, fname).replace("\\", "/")
                rows.append([full_path, label, "siteA"])
    return rows

# Build CSVs
for split in splits:
    rows = collect(split)
    df = pd.DataFrame(rows, columns=["path", "label", "site_id"])
    df.to_csv(f"{split}.csv", index=False)
    print(f"{split}.csv created with {len(df)} rows")