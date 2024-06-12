import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv2
from mtcnn import MTCNN
from model import *

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("./model/model.json", "./model/model_weights.h5")

face_detector_mtcnn = MTCNN()

ch = int(input("Press 0 for image, 1 for video, 2 for webcam:\n"))

if ch==1 or ch==0:
	path = input("Enter path:\n")
	# path = "disgust.jpeg"

if ch == 0 :
	frame = cv2.imread(path)
	# frame = cv2.resize(frame, (400,400))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# faces = face_cascade.detectMultiScale(gray, 1.3,4)
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	dict_faces = face_detector_mtcnn.detect_faces(frame_rgb) 
	faces =[]
	for dict_face in dict_faces:
		x,y,w,h = dict_face['box'][:]
		faces.append([x,y,w,h])

	print(faces)

	for (x,y,w,h) in faces:
		face = gray[y:y+h , x:x+h]
		roi = cv2.resize(face, (64,64))

		pred, confidence = model.predict_emotion(roi[np.newaxis, :, : , np.newaxis])

		cv2.putText(frame, pred + " (" + str(confidence) +")" , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
		cv2.rectangle(frame, (x,y), (x+w, y+h) , (0,255,255),2)
		print (pred)

	cv2.imshow("detection", frame)
	cv2.waitKey(0)
	

if ch == 1 :
	video = cv2.VideoCapture(path)

	while True:
		ret, frame = video.read()
		# loactions = face_recognition.face_locations(frame , model = "cnn")
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# faces = face_cascade.detectMultiScale(gray, 1.3,5)

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		dict_faces = face_detector_mtcnn.detect_faces(frame_rgb) 
		faces =[]
		for dict_face in dict_faces:
			x,y,w,h = dict_face['box'][:]
			faces.append([x,y,w,h])

		for (x,y,w,h) in faces:
			face = gray[y:y+h , x:x+h]
			roi = cv2.resize(face, (64,64))

			pred, confidence = model.predict_emotion(roi[np.newaxis, :, : , np.newaxis])

			cv2.putText(frame, pred + " (" + str(confidence) +")" , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
			cv2.rectangle(frame, (x,y), (x+w, y+h) , (0,255,255),2)

			# cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0) , 3)


		cv2.imshow("detection", frame)
		if cv2.waitKey(1) & 0xFF ==ord('q'):
			break

if ch == 2 :

	video = cv2.VideoCapture(0)
	
	while True:
		ret, frame = video.read()
		# loactions = face_recognition.face_locations(frame , model = "cnn")
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# faces = face_cascade.detectMultiScale(gray, 1.3,5)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		dict_faces = face_detector_mtcnn.detect_faces(frame_rgb) 
		faces =[]
		for dict_face in dict_faces:
			x,y,w,h = dict_face['box'][:]
			faces.append([x,y,w,h])

		for (x,y,w,h) in faces:
			face = gray[y:y+h , x:x+h]
			roi = cv2.resize(face, (64,64))

			pred, confidence = model.predict_emotion(roi[np.newaxis, :, : , np.newaxis])

			cv2.putText(frame, pred + " (" + str(confidence) +")" , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
			cv2.rectangle(frame, (x,y), (x+w, y+h) , (0,255,255),2)

			# cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0) , 3)

		cv2.imshow("detection", frame)
		if cv2.waitKey(1) & 0xFF ==ord('q'):
			break