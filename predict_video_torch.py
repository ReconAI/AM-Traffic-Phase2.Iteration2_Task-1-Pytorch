# import the necessary packages
from __future__ import print_function
import argparse
import json
from collections import deque
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized model")
ap.add_argument("-w", "--weights", required=True,
                help="path to the parameters of the model")
ap.add_argument("-i", "--input", required=True,
                help="path to our input video")
ap.add_argument("-l", "--labels", required=True,
                help="path to the labels")
ap.add_argument("-o", "--output", required=True,
                help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=1,
                help="size of queue for averaging")
args = vars(ap.parse_args())

# load dictionary of labels
print("[INFO] loading labels...")
json_file = open(args['labels'])
label_dict = json.load(json_file)

# load the trained model
print("[INFO] loading model...")
loc = torch.load(args["weights"], map_location=device)
model = torch.load(args["model"], map_location=device)
model.load_state_dict(loc["model"])
model.eval()

data_transforms = {
    'val': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}

def predict(image, model):
    # Pass the image through our model
    image_tensor = data_transforms['val'](image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    images = Variable(image_tensor)
    image_tensor = images.to(device)
    torch.no_grad()
    predict = F.softmax(model(image_tensor))
    print(predict.detach().cpu().numpy())
    return predict.cpu().detach().numpy()

# initialize the image mean for mean subtraction along with the
# predictions queue
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # make predictions on the frame and then update the predictions
    # queue
    preds = predict(frame, model)
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = label_dict[str(i)]

    # draw the condition on the output frame
    text = "{}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (0, 255, 0), 5)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # write the output frame to disk
    writer.write(output)

    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
