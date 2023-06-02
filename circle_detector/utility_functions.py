import numpy as np
import cv2
import imutils
import os
from tensorflow.keras.models import load_model
from django.conf import settings

input_image_width = 350
confThreshold = 0.9
probThreshold_nonc = 0.2
iouThreshold = 0.2
model = False

###

def load_my_model():
    global model
    if model == False:
        model_path = os.path.join(settings.MODELS, "seg_cnn_softmax.h5") 
        try:
            model = load_model(model_path)
            print(" => model loaded successfully!")
            return True
        except:
            return False
        # except Exception as e:
            # return e
    else:
        print(" => model is already loaded!")
        return True
###

def pre_process2(image):
  # try:
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
  thresh = cv2.threshold(image,np.mean(image),255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
  image = cv2.dilate(cv2.Canny(thresh, 0, 255), None) 
  ex = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
  if len(ex) > 0:
    cnt = sorted(ex, key=cv2.contourArea)[-1]
  else:
    return thresh
  mask = np.zeros((224, 224), np.uint8)
  masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
  masked = np.asarray(masked, dtype=np.float32)
  return masked

def pre_process(image):
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
  image = cv2.GaussianBlur(image, (5, 5), 0)
  if(image.shape[0] > 216):
      image = cv2.resize(image, (216, 216), interpolation=cv2.INTER_LINEAR)
  else:
      image = cv2.resize(image, (216, 216), interpolation=cv2.INTER_AREA)
  image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_REPLICATE)
  return image

###

def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image
	while True:
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):	#height
		for x in range(0, image.shape[1], stepSize):	#width
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                        
###

def genBoxes(img, stepSize = 20, scale = 1.5, winSize = (60, 60)):
  window_size = winSize
  boxes = []
  while(window_size[0] <= img.shape[0] and window_size[1] <= img.shape[1]):
    print("winSize, img.shape : ", window_size, img.shape)
    for(x, y, window) in sliding_window(img, stepSize, window_size):
      if(window.shape[0] != window_size[0] or window.shape[1] != window_size[1]):
        continue
      window = pre_process2(window)
      result = model.predict(np.array([window]))[0]
      if(result[0] >= confThreshold):
        boxes.append([x, y, x + window_size[0], y + window_size[1], result[0]])
    window_size = (int(window_size[0] *1.5), int(window_size[1]*1.5))
    stepSize = int(stepSize*1.5)
  return boxes


### IoU calculation

def area(rect):
  return (rect[2] - rect[0])*(rect[3] - rect[1])

def iou(rectA, rectB):
  rectRes = [0,0,0,0]
  rectRes[0] = max(rectA[0], rectB[0])
  rectRes[2] = min(rectA[2], rectB[2])
  rectRes[1] = max(rectA[1], rectB[1])
  rectRes[3] = min(rectA[3], rectB[3])
  intersection = area(rectRes)
  union = area(rectA) + area(rectB) - intersection # A U B = A + B - (AnB)
  if(union == 0):
    return 1
  iou = intersection / union
  print(union, intersection)
  return iou

###


def filterBoxes(boxes):
  boxes = sorted(boxes, key = lambda x : x[4], reverse = True)
  filteredBoxes = []
  while boxes :
    currBox = boxes.pop(0)
    boxes = [
        box for box in boxes
        if(iou(currBox[:4], box[:4]) < iouThreshold)
    ]
    filteredBoxes.append(currBox)
  return filteredBoxes

###


def circle_detector(img, image_name):
    if load_my_model() == False:
        return (False, "error occured while loading model!")
    img = cv2.resize(img, (input_image_width, int(input_image_width * (img.shape[0] / img.shape[1]))), interpolation=cv2.INTER_LINEAR)
    img_copy = img.copy()
    boxes = genBoxes(img, stepSize = 20, scale = 1.5, winSize = (90, 90))
    filtered_boxes = filterBoxes(boxes)
    for box in filtered_boxes:
        img_copy = cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
    for box in boxes:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
    cv2.imwrite(os.path.join(settings.BASE_DIR, "media", 'result-' + image_name), img_copy)
    print(len(boxes), len(filtered_boxes))
    return (True, img_copy)