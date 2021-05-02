from mobilenet_rm_filt_tf import MobileNetv1, remove_channel
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_model_optimization as tfmot
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from Prune import prune_channel_frac, show_l1_graphs, quantize
import time
import traceback


def save_model(trainedModel, fraction, specialName="", optim=0):
    if trainBeforeRemoval:
        modelName = "secretSauceFraction{}".format(fraction)
    else:
        modelName = "noSauceFraction{}".format(fraction)

    if not specialName == "":
        modelName = specialName
    converter = tf.lite.TFLiteConverter.from_keras_model(trainedModel)
    if optim:
        #converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative_dataset_gen
        modelName = modelName + "_optim"
    tflite_model = converter.convert()
    # Save the model.
    with open(f'savedModels/TFLite/' + modelName + '.tflite', 'wb') as f:
        f.write(tflite_model)


def representative_dataset_gen():
    for i in range(10):
        yield [np.random.uniform(low=0.0, high=1.0, size=(1, 32, 32, 3)).astype(np.float32)]


def instatiate_enviornment():
    global config
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # TODO: Convert the datasets to contain only float values
    X_train = X_train.astype('float32')
    test_images = X_test.astype('float32')
    # TODO: Normalize the datasets
    X_train = X_train / 255.0
    test_images = X_test / 255.0
    # TODO: Encode the labels into one-hot format
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    model = MobileNetv1()
    model.load_weights("mbnv1_tf.ckpt")
    # model.summary()
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])

    try:
        _, baseline_model_accuracy = model.evaluate(X_test, y_test)
    except Exception:
        traceback.print_exc()
    print("baseline model accuracy: " + str(baseline_model_accuracy))

    return model, X_train, y_train, test_images, y_test


tf.random.set_seed(42)
batch_size = 64
model, X_train, y_train, test_images, y_test = instatiate_enviornment()
show_l1_graphs(model)


saveModel = 1
frac = 0.7
trainBeforeRemoval = 0

special_name = "frac0.7"


model = prune_channel_frac(model, frac)
show_l1_graphs(model)
if trainBeforeRemoval:
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(test_images, y_test))
show_l1_graphs(model)
model = remove_channel(model)
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
show_l1_graphs(model)


time.sleep(3)


model.summary()
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=5,
                      validation_data=(test_images, y_test) )
show_l1_graphs(model)
##

if saveModel:
    save_model(model, frac, special_name, optim=0)



