import torch
from unets.resunet import Res18_UNet
from torch import nn
from torch.nn import functional as F

model = Res18_UNet(n_classes=2, layer=4)


class Out_model(nn.Module):
    def __init__(self):
        super(Out_model, self).__init__()
        self.model = Res18_UNet(n_classes=2, layer=4)
        self.model.load_state_dict(torch.load("checkpoints/network_state/acc94.74_model.pth"))

    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        x = x.squeeze(dim=0)[1]
        return x


if __name__ == "__main__":
    net = Out_model()
    example = torch.rand(1, 3, 128, 768)
    y = net(example)
    print(y.shape)
    trace_script_module = torch.jit.trace(net, example)
    trace_script_module.save("model.pt")


