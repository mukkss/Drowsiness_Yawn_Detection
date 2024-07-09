from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Event
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
import signal

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    stop_event.set()
    cv2.destroyAllWindows()
    vs.stop()
    sys.exit(0)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 100
YAWN_THRESH = 18
alarm_status = False
alarm_status2 = False
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if detector.empty():
    raise IOError("Unable to load the face detector XML file")

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

stop_event = Event()
signal.signal(signal.SIGINT, signal_handler)

while True:
    if stop_event.is_set():
        break

    frame = vs.read()
    if frame is None:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]
        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        drowsiness_alert = False
        yawn_alert = False

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                drowsiness_alert = True
                if not alarm_status:
                    alarm_status = True
                    print("DROWSINESS ALERT! Wake up the student!")
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            yawn_alert = True
            if not alarm_status2:
                alarm_status2 = True
                print("YAWN ALERT! Classes are getting boring, make some fun!")
        else:
            alarm_status2 = False

        if drowsiness_alert and yawn_alert:
            cv2.putText(frame, "DROWSINESS AND YAWN ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif drowsiness_alert:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif yawn_alert:
            cv2.putText(frame, "YAWN ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    # Perform cleanup tasks here
    stop_event.set()  # Set the Event to signal termination
    cv2.destroyAllWindows()  # Close OpenCV windows
    vs.stop()  # Stop the VideoStream
    sys.exit(0)  # Exit the program

signal.signal(signal.SIGINT, signal_handler)

cv2.destroyAllWindows()
vs.stop()