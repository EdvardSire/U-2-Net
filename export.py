import torch
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from pathlib import Path

net = U2NET(3,1)
netp = U2NETP(3,1)

path = Path("saved_models")
net.load_state_dict(torch.load((path / "u2net" / "u2net.pth"), map_location="cpu"))
netp.load_state_dict(torch.load((path / "u2netp" / "u2netp.pth"), map_location="cpu"))

models = [net, netp]


input = torch.randn(1,3, 320, 320)
for model in models:
    torch.onnx.export(model,
                      input,
                      f"{model.__class__.__name__}.onnx",
                      input_names=["input"],
                      output_names=["output"])
