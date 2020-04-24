# Import libraries
import os
import argparse
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import onnx2keras
from onnx2keras import onnx_to_keras
import onnx

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized model")
ap.add_argument("-w", "--weights", required=True,
                help="path to the parameters of the model")
ap.add_argument("-k", "--keras_path", required=True,
                help=" path we want save our converted keras model")


args = vars(ap.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Loading the model
loc = torch.load(args["weights"])
model = torch.load(args["model"])
model.load_state_dict(loc["model"])
model.eval()

# Convert and export to ONNX format
dummy_input = torch.ones((1, 3, 224, 224)).cuda()
path_onnx = os.path.join(args['keras_path'],
                         os.path.basename(args['model']).split('.')[0]+'.onnx')
torch.onnx.export(model, dummy_input, path_onnx,
                  input_names=['test_input'], output_names=['test_output'])

# Convert ONNX to Keras
onnx_model = onnx.load(path_onnx)
k_model = onnx_to_keras(onnx_model, ['test_input'])

keras.models.save_model(k_model, os.path.basename(args['model']).split('.')[0]+'.h5',
                        overwrite=True, include_optimizer=True)
