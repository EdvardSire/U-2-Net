import os
import sys
from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
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
    cv2.resizeWindow(name, 600, 600)
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

    
    #model_path = Path("u2netu2net_bce_itr_200000_train_0.191023_tar_0.016977.pth")
    model_path = Path("runs/SUAS_U2NET_2/model_name_bce_itr_200000_train_0.084_tar_0.001.pth")
    image_paths = list(Path("training_data/cutout_dumps").glob("*png"))

    test_salobj_dataset = SalObjDataset(image_paths = image_paths,
                                        mask_paths = [],
                                        transform=transforms.Compose([RescaleT(200),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    net = U2NET(3, 1) if(model_name=='u2net') else U2NETP(3,1)

    if torch.cuda.is_available():
        #net.load_state_dict(torch.load(("saved_models" / model_path)))
        net.load_state_dict(torch.load((model_path)))
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

        # save results to test_results folder
        predict_np = pred.cpu().data.numpy()
        show(cv2.resize(cv2.imread(image_paths[i].__str__()), (200, 200)))
        show(predict_np.reshape(200, 200))
        

        # save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
