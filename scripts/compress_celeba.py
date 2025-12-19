import sys

import torch
from torchvision import datasets, transforms
from tqdm import tqdm

pixel_range = 255
transform = transforms.Compose(
    [
        transforms.CenterCrop((140, 140)),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (255 * x).long()),
    ]
)

splits = ["train", "valid", "test"]

for split in splits:
    print(f"Processing split: {split}")

    dataset = datasets.CelebA(
        sys.argv[1], split=split, download=False, transform=transform
    )

    images = []

    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        img_uint8 = (img * 255).to(torch.uint8)
        images.append(img_uint8)

    images = torch.stack(images)  # shape (N,C,H,W)

    torch.save(images, f"datasets/celeba_{split}.pt")

    print(f"Saved celeba_{split}.npz with shape {images.shape}")
