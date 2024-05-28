import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 400, 400)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class RandomNoise(torch.nn.Module):
    def __init__(self, p = 0.5, sigma = 1.0):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def forward(self, x):
        return x + (torch.rand(1) <= self.p) * torch.randn_like(x) * self.sigma

class RandomDeleteRows(torch.nn.Module):
    def __init__(self, p = 0.5, mu = 5, sigma = 1):
        super().__init__()
        self.p = p
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        if torch.rand(1) > self.p:
            return x

        image, mask = x

        image_rows = image.shape[-2]
        num_delete = min(max(0, int(self.sigma * torch.randn(1) + self.mu)), image_rows - 1)
        row = int(torch.rand(1) * (image_rows - num_delete))

        return (
            torch.cat([
                image[:, :row, :],
                image[:, row+num_delete:, :],
            ], dim=1),
            torch.cat([
                mask[:, :row, :],
                mask[:, row+num_delete:, :],
            ], dim=1)
        )


class SalObjDataset(Dataset):
    def __init__(self,image_paths,mask_paths,transform=torch.nn.ModuleList()):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        images = cv2.imread(self.image_paths[idx].__str__())

        if self.mask_paths == []:
            labels = np.zeros([*images.shape[:-1], 1], dtype=np.uint8) # only for inference
        else:
            labels = cv2.imread(self.mask_paths[idx].__str__(), cv2.IMREAD_GRAYSCALE)
            labels = labels[:,:,np.newaxis]

        assert 3==len(images.shape) and 3==len(labels.shape)
        assert images.shape[2] == 3 and labels.shape[2] == 1

        for transform in self.transform:
            if isinstance(transform, transforms.RandomPhotometricDistort) or \
               isinstance(transform, transforms.Normalize) or \
               isinstance(transform, transforms.RandomInvert) or \
               isinstance(transform, RandomNoise):
                images = transform(images)
                continue 
            elif isinstance(transform, transforms.Resize):
                images = transform(images)
                labels = transform(labels)
            else:

                images, labels = transform((images, labels))

        # rip v2.resize, goes over 1.000
        return images, torch.clamp(labels, min=0, max=1) # pyright: ignore

