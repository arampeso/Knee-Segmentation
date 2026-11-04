from pathlib import Path
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
import torch

# -------------------- Step 1: Create dataset.csv --------------------
# folders 
xray_folder = Path("data/xrays")
mask_folder = Path("data/masks")
csv_folder = Path("data/csv")
csv_folder.mkdir(parents=True, exist_ok=True)

xrays = []
masks = []

for xray_path in natsorted(xray_folder.iterdir()): # sort xrays
    
    mask_path = mask_folder / xray_path.name

    xrays.append(str(xray_path))

    if mask_path.exists():
        masks.append(str(mask_path))
    else:
        # test xrays w/o masks
        masks.append("None")

# create df with the lists
df = pd.DataFrame({"xrays": xrays, "masks": masks})

# save to csv
df.to_csv(csv_folder / "dataset.csv", index=False)

print("Done! Amount of Xrays:", len(df))
print("Without Masks:", (df["masks"] == "None").sum())


# -------------------- Step 2: train/val/test split --------------------

# keep_default_na=False prevents the string "None" from being converted to NaN
df = pd.read_csv(f"{csv_folder}/dataset.csv", keep_default_na=False)

# Test set: all images without masks
test_df = df[df["masks"] == "None"]

# Remaining images: images with masks
labeled_df = df[df["masks"] != "None"]

# Split labeled images into 80% training and 20% validation (randomly)
train_df = labeled_df.sample(frac=0.8, random_state=59)
val_df = labeled_df.drop(train_df.index)

# Save the datasets as separate CSV files
train_df.to_csv(f"{csv_folder}/train.csv", index=False)
val_df.to_csv(f"{csv_folder}/val.csv", index=False)
test_df.to_csv(f"{csv_folder}/test.csv", index=False)

# -------------------- # Step 3: Create a bar chart --------------------

names = ["train", "val", "test"]
counts = [len(train_df), len(val_df), len(test_df)]

plt.bar(names, counts)
plt.title("Number of Samples per Dataset Split")
plt.ylabel("Number of Samples")
plt.savefig(f"{csv_folder}/split_check.png")

print("train:", len(train_df), "| val:", len(val_df), "| test:", len(test_df))