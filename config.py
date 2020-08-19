from tensorflow.keras.models import load_model
import cv2
import os

FACE_LOC = "face_detector"
MODEL = "mask_detector.model"
CONFIDENCE = 0.5

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([FACE_LOC, "deploy.prototxt"])
weightsPath = os.path.sep.join([FACE_LOC, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(MODEL)
