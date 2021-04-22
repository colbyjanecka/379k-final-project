import onnx
from onnxruntime.quantization import quantize_qat, QuantType
import glob

modelFilenames = []
for name in glob.glob('models/onnx/*'):
    modelFilenames.append(name)

for model_pth in modelFilenames:
    model_quant = "models/quant/" + model_pth.split('\\')[1]
    print(model_quant)

    quantized_model = quantize_qat(model_pth, model_quant)
'''
for model_pth in modelFilenames:
    model_fp32 = 'path/to/the/model.onnx'
    model_quant = 'path/to/the/model.quant.onnx'
    
    quantized_model = quantize_qat(model_fp32, model_quant)
'''