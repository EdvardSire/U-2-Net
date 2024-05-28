import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from pathlib import Path

import cv2
import numpy as np

from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

from binarize_shape import bgr2binarizedshape

def show(img, name: str|list = "window"):
    if type(img) != list:
        img = [img]

    if type(name) != list:
        name = [name]

    for i, im in enumerate(img):
        winname = f"{name[0]}_{i}" if i >= len(name) else name[i]
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL) #pyright: ignore
        cv2.resizeWindow(winname, 600, 600)#pyright: ignore
        cv2.moveWindow(winname, 640* (i % 3), 20 + 640 * (i //3)) #pyright: ignore
        cv2.imshow(winname, im)#pyright: ignore
    cv2.waitKey(0)#pyright: ignore
    cv2.destroyAllWindows()#pyright: ignore


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def main():
    model_name='u2net'#u2netp

    IMSIZE = (200, 200)

    
    #model_path = Path("u2netu2net_bce_itr_200000_train_0.191023_tar_0.016977.pth")
    model_3_path = Path("suas_u2net_3_itr199500.pth")
    model_2_path = Path("suas_u2net_2_itr198500.pth")
    image_paths = list(Path("training_data/cutout_dumps").glob("*png"))

    test_salobj_dataset = SalObjDataset(image_paths = image_paths,
                                        mask_paths = [],
                                        transform=torch.nn.ModuleList([
                                            transforms.ToImage(), # hwc -> chw
                                            transforms.ToDtype(torch.float32, scale=True),
                                            #transforms.RandomApply(torch.nn.ModuleList([
                                            #    transforms.ElasticTransform(50.0, 3.5)]), p=0.3),
                                            #transforms.RandomPhotometricDistort(),
                                            #transforms.RandomInvert(p=0.2),
                                            #transforms.RandomAdjustSharpness(sharpness_factor=0.3),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            transforms.Resize(IMSIZE)
                                        ]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    net3 = U2NET(3, 1) if(model_name=='u2net') else U2NETP(3,1)
    net2 = U2NET(3, 1)

    if torch.cuda.is_available():
        #net.load_state_dict(torch.load(("saved_models" / model_path)))
        net3.load_state_dict(torch.load((model_3_path)))
        net3.cuda()
        net2.load_state_dict(torch.load((model_2_path)))
        net2.cuda()
    else:
        net3.load_state_dict(torch.load(("saved_models" / model_3_path), map_location='cpu'))
    net3.eval()

    for i, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",image_paths[i].name)

        inputs_test = data_test[0]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        
        sys.stdout.flush()
        d1,d2,d3,d4,d5,d6,d7= net3(inputs_test)

        # normalization
        pred3 = d1[:,0,:,:]
        pred3 = normPRED(pred3)

        d1,d2,d3,d4,d5,d6,d7= net2(inputs_test)
        pred2 = d1[:,0,:,:]
        pred2 = normPRED(pred2)


        predict_np_3 = pred3.cpu().data.numpy()
        predict_np_2 = pred2.cpu().data.numpy()

        orig = cv2.imread(str(image_paths[i]))

        binarized = bgr2binarizedshape(orig)
        binarized = cv2.resize(binarized, IMSIZE)
        show([
            cv2.resize(orig, IMSIZE),#pyright: ignore
            predict_np_2.reshape(IMSIZE),
            predict_np_3.reshape(IMSIZE),
            #cv2.threshold((predict_np_3.reshape(IMSIZE) * 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)[1],
            binarized
        ], ["original", "SUAS_2", "SUAS_3", "Binarized"])
        #predict = predict_np.reshape(IMSIZE)
        #predict = cv2.normalize(predict, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # pyright: ignore
        #_, predict = cv2.threshold(predict, 0, 255, cv2.THRESH_OTSU)
        #show(predict)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
