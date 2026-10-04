"""
Real-Time Face Mask Detection using OpenCV and TensorFlow/Keras
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet, min_confidence=0.5):
    # Grab dimensions and construct blob for OpenCV Caffe face detector (300x300)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Forward pass through face detector
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Run predictions if any face detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prototxtPath = os.path.join(base_dir, "face_detector", "deploy.prototxt")
    weightsPath = os.path.join(base_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    maskModelPath = os.path.join(base_dir, "mask_detector.model.h5")

    print("[INFO] Loading face detector model...")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("[INFO] Loading face mask detector model...")
    maskNet = load_model(maskModelPath)

    print("[INFO] Starting video stream (press 'q' in the window to quit)...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    try:
        while True:
            frame = vs.read()
            if frame is None:
                print("[WARNING] Could not read frame from camera.")
                break

            frame = imutils.resize(frame, width=800)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                confidence = max(mask, withoutMask) * 100
                labelText = f"{label}: {confidence:.2f}%"

                cv2.putText(frame, labelText, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Face Mask Detector - Press Q to Quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        print("[INFO] Cleaning up...")
        cv2.destroyAllWindows()
        vs.stop()

if __name__ == "__main__":
    main()
