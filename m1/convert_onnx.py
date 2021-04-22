import torch
from models.pytorch import mobilenet_pt
import onnx
import onnxruntime
import keras2onnx

def export_pt_to_onnx(model, outputFileName):
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False)
    outputPath = "models/onnx/" + outputFileName + ".onnx"
    torch.onnx.export(model, dummy_input, outputPath,  export_params=True,  opset_version=10)

    
fractions = [0.05, 0.25, 0.5, 0.75, 0.9]
epochs=[0,3,5]


for frac in fractions:
    for epoch in epochs:
        model_name = "frac_" + str(frac) + "_epochs_" + str(epoch)
        print("Converting at ", frac, " fractions and ", epoch, " epochs.")

        model = mobilenet_pt.MobileNetv1()
        model.load_state_dict(torch.load(("models/pytorch/" + model_name + ".pt"), map_location=torch.device('cpu')))
        export_pt_to_onnx(model, "mbnv1_pt")
