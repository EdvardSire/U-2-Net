import cv2
import numpy as np
from inference import show
from pathlib import Path


if __name__ == "__main__":
    train_path = Path("training_data/train")
    images = list((train_path / "images").glob("*jpg"))
    masks = list((train_path  / "masks").glob("*png"))
    masks = list(Path("../suas24_classification_benchmark/cutout_dumps").glob("*otsu.png"))
    
    for image_path, mask_path in zip(images, masks):
        mask = cv2.imread(mask_path.__str__())
        mask = cv2.copyMakeBorder(mask, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0) # pyright: ignore
        gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=100,
                qualityLevel=0.1,
                minDistance=15,
                blockSize=15,
                useHarrisDetector=True)
                
        try:
            corners = np.int0(corners) # pyright: ignore
            for i in corners:
                 x,y = i.ravel()
                 cv2.circle(mask,(x,y),5,(0,255, 0))

            show(mask)
        except:
            pass

