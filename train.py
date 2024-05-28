from skimage.color import lab2lch
import torch
import cv2
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

from pathlib import Path

from torchvision.transforms import v2 as transforms
from data_loader import SalObjDataset, RandomNoise, RandomDeleteRows

from model import U2NET
from model import U2NETP


def show(img, name = "window"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 800)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss



def train(dataloader,
          epoch_num,
          batch_size_train,
          train_num, optimizer,
          running_loss,
          running_tar_loss,
          ite_num4val,
          save_frequency,
          writer: SummaryWriter | None = None
          ):

    step = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, (inputs, labels) in enumerate(dataloader):
            step = step + 1
            ite_num4val = ite_num4val + 1

            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs.type(torch.float32).cuda())
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels.type(torch.float32).cuda())

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            if writer:
                writer.add_scalar("Loss/train", loss, step)
                writer.flush()

            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, step, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if step % save_frequency == 0:

                torch.save(net.state_dict(),
                           Path(writer.get_logdir()) / f"model_name_bce_itr_{step}_train_{running_loss / ite_num4val:.3f}_tar_{running_tar_loss / ite_num4val:.3f}.pth")
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

if __name__ == "__main__":

    model_name = 'u2net' #u2netp
    model_dir = Path("saved_models") / model_name
    net = U2NET(3, 1) if(model_name=='u2net') else U2NETP(3,1)
    if torch.cuda.is_available():
        net.cuda()

    train_images = list((Path("training_data") / "SUAS" / "images").glob("*png"))
    train_masks = [Path(path.__str__().replace("images", "shape_masks")) for path in train_images]
    assert len(train_images) == len(train_masks)
    assert train_images[0].exists() and train_masks[0].exists()


    epoch_num = 100000
    batch_size_train = 32
    train_num = len(train_images)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    bce_loss = nn.BCELoss(size_average=True)
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frequency = 500

    dataset = SalObjDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        transform=torch.nn.ModuleList([
            transforms.ToImage(), # hwc -> chw
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.ElasticTransform(50.0, 4.0)]), p=0.3),
            transforms.RandomPhotometricDistort(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.3),
            RandomNoise(p=0.5, sigma=0.1),
            RandomDeleteRows(p=0.5, mu=5, sigma=20),
            RandomDeleteRows(p=0.5, mu=5, sigma=20),
            RandomDeleteRows(p=0.5, mu=5, sigma=20),
            RandomDeleteRows(p=0.5, mu=5, sigma=20),
            transforms.Resize((200,200)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    # for images, labels in dataloader:
    #     inputs = images.numpy()[0].transpose(1,2,0)
    #     labels = labels.numpy()[0].transpose(1,2,0)
    #     print(inputs.shape, labels.shape)
    #     assert inputs.shape[:2] == (200, 200)
    #     assert labels.shape[:2] == (200, 200)
    #     #show(inputs)
    #     #show(labels)
    # exit()

    LOGDIR=Path("runs")
    LOGDIR.mkdir(exist_ok=True)
    experiment_name = "SUAS_" + net.__class__.__name__
    paths = [path for path in LOGDIR.iterdir() if path.name.startswith(experiment_name)]
    try:
        iternum = 1+int(max([iternum.__str__().split("_")[-1] for iternum in paths]))
    except:
        iternum = 1

    writer = SummaryWriter(log_dir=f"runs/{experiment_name}_{iternum}")
    train(dataloader=dataloader,
    epoch_num=epoch_num,
    batch_size_train = batch_size_train,
    train_num = train_num,
    optimizer = optimizer,
    running_loss = running_loss,
    running_tar_loss = running_tar_loss,
    ite_num4val = ite_num4val,
    save_frequency = save_frequency,
    writer=writer
    )
