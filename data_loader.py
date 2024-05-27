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
            labels = np.zeros((1,1,1), dtype=np.uint8) # only for inference
        else:
            labels = cv2.imread(self.mask_paths[idx].__str__(), cv2.IMREAD_GRAYSCALE)
            labels = labels[:,:,np.newaxis]

        assert 3==len(images.shape) and 3==len(labels.shape)
        assert images.shape[2] == 3 and labels.shape[2] == 1

        for transform in self.transform:
            if isinstance(transform, transforms.RandomPhotometricDistort) or \
               isinstance(transform, transforms.Normalize) or \
               isinstance(transform, transforms.RandomInvert):
                images = transform(images)
                continue 

            images, labels = transform((images, labels))

        # rip v2.resize, goes over 1.000
        return images, torch.clamp(labels, min=0, max=1) # pyright: ignore

