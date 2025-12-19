from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def merge_npz_to_pt(folder_path, output_path):
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob("*.npz"))

    imgs_list = []
    labels_list = []

    for f in tqdm(files):
        data = np.load(f)
        imgs = torch.from_numpy(data["data"])  # (N,C,H,W)
        imgs_list.append(imgs)
        if "labels" in data.files:
            labels = torch.from_numpy(data["labels"])
            labels_list.append(labels)

    all_imgs = torch.cat(imgs_list, dim=0)
    if labels_list:
        all_labels = torch.cat(labels_list, dim=0)
        torch.save({"image": all_imgs, "labels": all_labels}, output_path)
    else:
        torch.save({"image": all_imgs}, output_path)
    print(f"Saved {output_path} with shape {all_imgs.shape}")


merge_npz_to_pt("Imagenet32_train_npz", "imagenet32_train.pt")
merge_npz_to_pt("Imagenet32_val_npz", "imagenet32_val.pt")
merge_npz_to_pt("Imagenet64_train_npz", "imagenet64_train.pt")
merge_npz_to_pt("Imagenet64_val_npz", "imagenet64_val.pt")
