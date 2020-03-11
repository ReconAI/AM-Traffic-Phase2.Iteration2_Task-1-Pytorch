# USAGE
# python3 predict_images_torch.py --model ./models/pytorch/weather_model.pt --weights  ./models/pytorch/weights_weather.pth --input ./input --output ./output_weather_torch --output ./output_weather_torch --labels ./weather_labels.json
# python3 predict_images_torch.py --model ./models/pytorch/road_model.pt --weights  ./models/pytorch/weights_road.pth --input ./input --output ./output_road_torch --labels ./road_labels.json

# import the necessary packages
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import argparse
import glob
import cv2
import os

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
loc = torch.load(args["weights"])
model = torch.load(args["model"])
model.load_state_dict(loc["model"])
model.eval()


data_transforms = {
    'val': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def predict(image, model):
    # Pass the image through our model
    image_tensor = data_transforms['val'](image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    images = Variable(image_tensor)
    image_tensor = images.to(device)
    image_tensor.cuda()
    torch.no_grad()
    predict = F.softmax(model(image_tensor))
    print(predict.detach().cpu().numpy())
    return predict.cpu().detach().numpy()



# loop over images
images = glob.glob(args['input']+'/*.jpg')
print("[INFO] Making predictions on images...")
for image in images:
	# clone the output image, then convert it from BGR to RGB
	# ordering, resize the image to a fixed 224x224, and then
	# perform preprocessing
	img = cv2.imread(image)
	output = img.copy()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# make predictions on the image
	preds = predict(img,model)
	i = np.argmax(preds)
	label = label_dict[str(i)]

	# draw the condition on the output image
	text = "{}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (255, 0, 0), 5)
	image_name = os.path.basename(image)
	# save the image in the output path
	cv2.imwrite(os.path.join(args['output'], image_name),output) 


print("[INFO] finished!!")