import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

from pathlib import Path

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP


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
          writer = SummaryWriter()
          ):

    step = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(dataloader):
            step = step + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            if writer:
                writer.add_scalar("Loss/train", loss, step)
            writer.flush()


            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, step, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if step % save_frequency == 0:

                torch.save(net.state_dict(), model_dir.__str__() + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (step, running_loss / ite_num4val, running_tar_loss / ite_num4val))
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
    save_frequency = 2000

    dataset = SalObjDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        transform=transforms.Compose([
            RescaleT(200),
            # RandomCrop(190),
            ToTensorLab(flag=0)]))
    dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    LOGDIR=Path("runs")
    LOGDIR.mkdir(exist_ok=True)
    experiment_name = "SUAS_" + net.__class__.__name__
    paths = [path for path in LOGDIR.iterdir() if path.name.startswith(experiment_name)]
    try:
        iternum = 1+int(max([iternum.__str__().split("_")[1] for iternum in paths]))
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

