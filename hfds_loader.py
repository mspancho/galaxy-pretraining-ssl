import datasets

class HFTwoCropsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, two_crops_transform):
        self.dataset = hf_dataset
        self.two_crops_transform = two_crops_transform  # expects [q, k] return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        if not hasattr(img, "convert"):
            if isinstance(img, torch.Tensor):
                img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(np.array(img))
        views = self.two_crops_transform(img)
        label = sample.get("label", 0)
        return views, label
