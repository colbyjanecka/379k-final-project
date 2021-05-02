from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import tflite_runtime.interpreter as tflite
import time
import copy
# TODO: add argument parser

# TODO: add one argument for selecting VGG or MobileNet-v1 models

# TODO: Modify the rest of the code to use the arguments correspondingly




def evaluate_model(tflite_model_name):
  print(tflite_model_name)
  # Get the interpreter for TensorFlow Lite model
  interpreter = tflite.Interpreter(model_path=tflite_model_name)
  # Very important: allocate tensor memory
  interpreter.allocate_tensors()
  # Get the position for inserting the input Tensor
  input_details = interpreter.get_input_details()
  print(input_details)
  # Get the position for collecting the output prediction
  output_details = interpreter.get_output_details()
  input_shape = (32,32)
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
  #interpreter.set_tensor(input_details[0]['index'], input_data)
  # Label names for CIFAR10 Dataset
  label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  reduction_param = 1
  correct = 0
  i = 0
  total_time = 0
  for filename in tqdm(os.listdir("test_deployment")):
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:

      input_image = np.expand_dims(np.uint8(img), axis=0)

      # Set the input tensor as the image
      interpreter.set_tensor(input_details[0]['index'], input_image)

      start_time = time.time()
      # Run the actual inference
      interpreter.invoke()

      total_time = total_time + (time.time() - start_time)
      # Get the output tensor
      pred_tflite = interpreter.get_tensor(output_details[0]['index'])

      # Find the prediction with the highest probability
      top_prediction = np.argmax(pred_tflite[0])

      # Get the label of the predicted class
      pred_class = label_names[top_prediction]

      trueLabel = filename.split('_')[1].split('.')[0]

      if pred_class == trueLabel:
        correct = correct + 1
      i = i + 1
  print(correct / (10000 / reduction_param) * 100)
  print("Total Time: {}".format(total_time) )
  print("Average Latency per Image: {}".format(total_time/10000*1000) )


tflite_model_name = "savedModels/TFLite/frac0.99_2_optim.tflite" # TODO: insert TensorFlow Lite model name

evaluate_model(tflite_model_name)