#import the necessary packages
from __future__ import print_function
import argparse
import time
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras

keras.backend.set_image_data_format("channels_first")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_node", required=True,
                help="name of the input node")
ap.add_argument("-o", "--output_node", required=True,
                help="name of the output node")
ap.add_argument("-p", "--path", required=True,
                help=" path to the trt model")
ap.add_argument("--img_path", required=True,
                help="path to the image we want to predict")
ap.add_argument("-l", "--labels", required=True,
                help="path to the labels")
ap.add_argument("-s", "--shape_size", required=True, type=int,
                help="input shape")
ap.add_argument("--time", type=str2bool, nargs='?',
                const=True, default=False,
                help="set it to True if you want to compute the execution time.")

args = vars(ap.parse_args())

# ----------------------------- Load TensorRT graph ----------------------------
input_names = [args['input_node']]
output_names = [args['output_node']]

json_file = open(args['labels'])
label_dict = json.load(json_file)

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph(args['path'])

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

# ------------------------------- Make predictions -------------------------------

img = image.load_img(args['img_path'], target_size=(args['shape_size'], args['shape_size']))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.resnet.preprocess_input(x)

feed_dict = {
    input_tensor_name: x
}
preds = tf_sess.run(output_tensor, feed_dict)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
i = np.argmax(preds)
label = label_dict[str(i)]
print('Predicted:', label)

if args['time']:
    times = []
    for i in range(20):
        start_time = time.time()
        one_prediction = tf_sess.run(output_tensor, feed_dict)
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))
