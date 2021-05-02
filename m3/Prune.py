from mobilenet_rm_filt_tf import MobileNetv1, remove_channel
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_model_optimization as tfmot
import tempfile
import matplotlib.pyplot as plt
import numpy as np


def prune_channel_frac(model, fraction):
    i = 0
    layerWeights = []
    layerIndex = []
    for layer in model.layers:
        if layer.name[:6] == "conv2d":
            layerWeights.append(layer.get_weights()[0])
            layerIndex.append(i)
        i = i + 1

    for i in range(len(layerWeights)):
        weight = layerWeights[i]
        dict = {}
        num_filters = len(weight[0, 0, 0, :])

        for j in range(num_filters):
            w_s = np.sum(abs(weight[:, :, :, j]))
            filt = 'filt_{}'.format(j)
            dict[filt] = w_s

        sorted_dict = sorted(dict.items(), key=lambda kv: kv[1])


        for j in range(num_filters):
            current_frac = (j + 1) / num_filters
            channel_num = sorted_dict[j][0].split("_")[1]

            if current_frac <= fraction:
                weight[:, :, :, int(channel_num)] = set_channel_weights_to_0(weight[:, :, :, int(channel_num)])
        layerWeights[i] = weight

        '''
        for j in range(num_filters):
            w_s = np.sum(abs(weight[:, :, :, j]))
            filt = 'filt_{}'.format(j)
            dict[filt] = w_s

        sorted_dict = sorted(dict.items(), key=lambda kv: kv[1])

        weights_value = []
        for elem in sorted_dict:
            weights_value.append(elem[1])

        xc = range(num_filters)
        plt.figure(i + 1, figsize=(7, 5))
        plt.plot(xc, weights_value)
        plt.xlabel("channel num")
        plt.ylabel("L1 Norm")
        plt.title("conv layer {}".format(i + 1))
        plt.grid(True)
        plt.show()
        '''

    for i in range(len(layerWeights)):
        idx = layerIndex[i]
        model.layers[idx].set_weights([layerWeights[i]])

    return model


def show_l1_graphs(model):
    i = 0
    layerWeights = []
    layerIndex = []
    for layer in model.layers:
        if layer.name[:6] == "conv2d":
            layerWeights.append(layer.get_weights()[0])
            layerIndex.append(i)
        i = i + 1


    for i in range(len(layerWeights)):
        if (i == 12):
            weight = layerWeights[i]
            dict = {}
            num_filters = len(weight[0, 0, 0, :])

            for j in range(num_filters):
                w_s = np.sum(abs(weight[:, :, :, j]))
                filt = 'filt_{}'.format(j)
                dict[filt] = w_s

            sorted_dict = sorted(dict.items(), key=lambda kv: kv[1])

            weights_value = []
            for elem in sorted_dict:
                weights_value.append(elem[1])

            xc = range(num_filters)
            plt.figure(i + 1, figsize=(7, 5))
            plt.plot(xc, weights_value)
            plt.xlabel("channel num")
            plt.ylabel("L1 Norm")
            plt.title("conv layer {}".format(i + 1))
            plt.grid(True)
            plt.show()

def set_channel_weights_to_0(channel):
    for i in range(channel[0][0].shape[0]):
        channel[0][0][i] = float(0.0)

    return channel

def quantize_channel(channel):
    for i in range(channel[0][0].shape[0]):
        channel[0][0][i] = np.uint8(channel[0][0][i])

    return channel

def quantize(model):
    i = 0
    layerWeights = []
    layerIndex = []
    for layer in model.layers:
        print(layer.name[:6], len(layer.get_weights()))
        if layer.name[:6] == "conv2d":

            layerWeights.append(layer.get_weights()[0])
            layerIndex.append(i)
        i = i + 1


    for i in range(len(layerWeights)):
        weight = layerWeights[i]
        dict = {}
        num_filters = len(weight[0, 0, 0, :])

        for j in range(num_filters):
            w_s = np.sum(abs(weight[:, :, :, j]))
            filt = 'filt_{}'.format(j)
            dict[filt] = w_s

        sorted_dict = sorted(dict.items(), key=lambda kv: kv[1])

        for j in range(num_filters):
            current_frac = (j + 1) / num_filters
            channel_num = sorted_dict[j][0].split("_")[1]


            weight[:, :, :, int(channel_num)] = quantize_channel(weight[:, :, :, int(channel_num)])
        layerWeights[i] = weight

    for i in range(len(layerWeights)):
        idx = layerIndex[i]
        model.layers[idx].set_weights([layerWeights[i]])

    return model


def quantize2(model):
    quant_aware_model = tfmot.quantization.keras.quantize_model(base_model)
