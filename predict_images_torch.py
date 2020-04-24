# import the necessary packages
from __future__ import print_function
import os
import argparse
import json
import glob
import torch
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
import numpy as np
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized model")
ap.add_argument("-w", "--weights", required=True,
                help="path to the parameters of the model")
ap.add_argument("-i", "--input", required=True,
                help="path to our input video")
ap.add_argument("-o", "--output", required=True,
                help="path to our output video")
ap.add_argument("-l", "--labels", required=True,
                help="path to the labels")

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
    return predict.cpu().detach().numpy()
y_true = []
y_pred = []
# loop over images in every category
categories = os.listdir(args['input'])
for cat in categories:
    images = glob.glob(os.path.join(args['input'], cat, '*.jpg'))
    print("[INFO] Making predictions on images...")
    true_label = label_dict[str(cat)]
    for image in images:
        # clone the output image, then convert it from BGR to RGB
        # ordering, resize the image to a fixed 224x224, and then
        # perform preprocessing
        img = cv2.imread(image)
        output = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # make predictions on the image
        preds = predict(img, model)
        i = np.argmax(preds)
        label = label_dict[str(i)]
        y_pred.append(label)
        y_true.append(true_label)

        # draw the condition on the output image
        text = "Predicted: {},\n Actual: {}".format(label, true_label)
        y0, dy = 50, 37
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(output, line, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
        image_name = os.path.basename(image)
        # save the image in the output path
        cv2.imwrite(os.path.join(args['output'], image_name), output)

print("Classification Report", classification_report(y_true, y_pred,
                                                     target_names=list(label_dict.values())))
print("Accuracy score", accuracy_score(y_true, y_pred))
print("[INFO] finished!!")
