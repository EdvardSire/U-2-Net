import os
import sys
from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 400, 400)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():
    model_name='u2net'#u2netp

    image_dir = Path("training_data") / "test" / "images"
    image_dir = Path("../suas24_classification_benchmark/test_data/")
    
    image_paths = list(image_dir.glob("*png"))



    test_salobj_dataset = SalObjDataset(image_paths = image_paths,
                                        mask_paths = [],
                                        transform=transforms.Compose([RescaleT(200),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    net = U2NET(3, 1) if(model_name=='u2net') else U2NETP(3,1)

    model_path = Path("first_suas_model.pth")
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(("saved_models" / model_path)))
        net.cuda()
    else:
        net.load_state_dict(torch.load(("saved_models" / model_path), map_location='cpu'))
    net.eval()

    for i, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",image_paths[i].name)

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        
        print(inputs_test.shape)
        sys.stdout.flush()
        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        predict_np = pred.cpu().data.numpy()
        show(cv2.imread(image_paths[i].__str__()))
        predict = predict_np.reshape(200,200)
        predict = cv2.normalize(predict, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # pyright: ignore
        print(predict)
        _, predict = cv2.threshold(predict, 0, 255, cv2.THRESH_OTSU)
        show(predict)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
