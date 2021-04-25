import numpy as np
import onnxruntime
from tqdm import tqdm
import os
import time
from PIL import Image
import argparse
import psutil
import telnetlib as tel
import gpiozero
from measurement import getTelnetPower


dict(psutil.virtual_memory()._asdict())
pre_inference_memory = psutil.virtual_memory().used/1000000
print('current memory usage (pre-inference):', psutil.virtual_memory().used/1000000, 'MB')


def evaluate_model(onnx_model_name):

    # Create Inference session using ONNX runtime
    sess = onnxruntime.InferenceSession("models/quant/" + onnx_model_name)

    # Get the input name for the ONNX model
    input_name = sess.get_inputs()[0].name
    print("Input name  :", input_name)

    # Get the shape of the input
    input_shape = sess.get_inputs()[0].shape
    print("Input shape :", input_shape)

    # Mean and standard deviation used for PyTorch models
    mean = np.array((0.4914, 0.4822, 0.4465))
    std = np.array((0.2023, 0.1994, 0.2010))

    # Label names for CIFAR10 Dataset
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    SP2_tel = tel.Telnet("192.168.4.1")
    total_power = 0
    max_memory = 0
    max_power = 0
    total_time = 0
    count = 0
    total_correct = 0
    true_start = time.time()
    framework = 'pt'
    cumulative_memory = 0

    # The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
    for filename in tqdm(os.listdir("test_deployment")):
        # Take each image, one by one, and make inference
        with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
            #print("Image shape:", np.float32(img).shape)

            if framework == 'pt':
            # For PyTorch models ONLY: normalize image
                input_image = (np.float32(img) / 255. - mean) / std
            # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
                input_image = np.expand_dims(np.float32(input_image), axis=0)

            if framework == 'tf':
            # For TensorFlow models ONLY: Add the Batch axis in the data Tensor (H, W, C)
                input_image = np.expand_dims(np.float32(img), axis=0)
                #print("Image shape after expanding size:", input_image.shape)

            if framework == 'pt':
            # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
                input_image = input_image.transpose([0, 3, 1, 2])

            # Get start time before inference
            start_time = time.time()

            # Run inference and get the prediction for the input image
            pred_onnx = sess.run(None, {input_name: input_image})[0]

            # Add inference time to total time
            total_time = total_time + (time.time() - start_time)

            # Find the prediction with the highest probability
            top_prediction = np.argmax(pred_onnx[0])

            # Get the label of the predicted class
            pred_class = label_names[top_prediction]

            true_label = filename.split('_')[1].split('.')[0]

            if pred_class == true_label:
                total_correct = total_correct + 1

            dict(psutil.virtual_memory()._asdict())
            cumulative_memory = cumulative_memory + (psutil.virtual_memory().used/1000000 - pre_inference_memory)
            count = count + 1


        # after inference, save the statistics for cpu usage and power consumption
        last_time = time.time()#time_stamp
        total_power = getTelnetPower(SP2_tel, total_power)
        if max_power < total_power:
            max_power = total_power
        
        current_memory_usage = psutil.virtual_memory().used/1000000
        if max_memory < current_memory_usage:
            max_memory = current_memory_usage
        
        #cpu_temp = gpiozero.CPUTemperature().temperature
        
    print(" ---------------------------------------------------------------------------------------------------------- ")
    print("accuracy: ", (total_correct/count*100))
    print("Total Inference Time: ", total_time)
    print("Max memory usage: ", max_memory)
    print("Average Latency per Image: ", total_time/10000)
    print("Max power consumption: ", max_power)
    print('inference memory usage:', cumulative_memory/count)
    print(" ---------------------------------------------------------------------------------------------------------- ")

    
fractions = [0.05, 0.25, 0.5, 0.75, 0.9, 0.976]
epochs = [0, 3, 5, 25, 100]

for frac in fractions:
    for epoch in epochs:
        model_name = "frac_" + str(frac) + "_epochs_" + str(epoch) + ".onnx"
        print("Evaluating at ", frac, " fractions and ", epoch, " epochs.")
        evaluate_model(model_name)
