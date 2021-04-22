import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import torchsummary
import time
import mobilenet_rm_filt_pt
import torch.nn.utils.prune as prune
from main_pt import train_model
from nni.algorithms.compression.pytorch.pruning import LevelPruner

parser = argparse.ArgumentParser(description='EE379K HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
args = parser.parse_args()

batch_size = args.batch_size

random_seed = 1
torch.manual_seed(random_seed)

train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)
test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def channel_fraction_pruning(model, fraction):
    for idx, m in enumerate(model.modules()):
        # print(idx, '->', m)
        i = 0
        for layer in m.children():
            name = str(layer).split("(")[0]

            if name == "Conv2d":
                if not int(str(layer).split("(")[1].split(",")[0]) == int(str(layer).split("(")[1].split(",")[1]):
                    #print(layer)
                    prune.ln_structured(layer, "weight", amount=fraction, n=1, dim=0)
                    prune.remove(layer, 'weight')



def export_pt_to_onnx(model, outputFileName):
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False)
    outputPath = "models/onnx/" + outputFileName + ".onnx"
    torch.onnx.export(model, dummy_input, outputPath, export_params=True, opset_version=11)

fractions = [0.05, 0.25, 0.5, 0.75, 0.9, 0.976]
epochs = [0, 3, 5, 25, 100]
model = mobilenet_rm_filt_pt.MobileNetv1()

for frac in fractions:
    for epoch in epochs:

        print(str(frac), str(epoch))
        model = mobilenet_rm_filt_pt.MobileNetv1()
        model = model.cuda()
        model.load_state_dict(torch.load("mbnv1_pt.pt", map_location=torch.device("cuda")))
        criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        channel_fraction_pruning(model, float(frac))

        pruned_model = mobilenet_rm_filt_pt.remove_channel(model)

        train_model(model, epoch, train_loader, test_loader, optimizer, criterion, len(train_dataset), batch_size)

        model.cpu()
        export_pt_to_onnx(model, "frac_" + str(frac) + "_epochs_" + str(epoch))
