# -*- coding: utf-8 -*-
"""
Used for converting the
keras model into trt
that can be used on Jetson nano
"""
# Import the needed libraries
from __future__ import print_function
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

tf.keras.backend.set_learning_phase(0)
#use this if we have batch norm layer in our network

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trt_path", required=True,
                help=" path we want save our converted models")
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized keras model")
ap.add_argument("-o", "--output_node", required=True,
                help="name of the output node")

args = vars(ap.parse_args())

# path we wanna save our converted TF-model
MODEL_PATH = args['trt_path']

# load the Keras model
model = load_model(args['model'])

# save the model to Tensorflow model
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, MODEL_PATH+"/tf_model")
print("Keras model is successfully converted to TF model in "+MODEL_PATH)

# has to be use this setting to make a session for TensorRT optimization
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # import the meta graph of the tensorflow model
    saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, "tf_model.meta"))
    # then, restore the weights to the meta graph
    saver.restore(sess, MODEL_PATH+"/tf_model")
    # specify which tensor output you want to obtain
    # (correspond to prediction result)
    your_outputs = [args['output_node']]
    # convert to frozen model
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, # session
        tf.get_default_graph().as_graph_def(), # graph+weight from the session
        output_node_names=your_outputs)
    #write the TensorRT model to be used later for inference
    with gfile.FastGFile(os.path.join(MODEL_PATH, "frozen_model.pb"), 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")
# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,# frozen model
    outputs=your_outputs,
    max_batch_size=1,# specify your max batch size
    max_workspace_size_bytes=1*(10**9),# specify the max workspace
    precision_mode="FP32") # precision, can be "FP32" (32 floating point precision) or "FP16"

#write the TensorRT model to be used later for inference
with gfile.FastGFile(os.path.join(MODEL_PATH, "tensorrt_model.pb"), 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")

# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)
