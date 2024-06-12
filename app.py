from model import *
import streamlit as st
from PIL import Image, ImageEnhance
from mtcnn import MTCNN

st.set_option('deprecation.showfileUploaderEncoding', False)


def enhancer(img, enhance_type, i):

	flag = False

	if enhance_type == 'Contrast':

		c_rate = st.sidebar.slider("Contrast",0.5,3.5, 1.2)
		enhancer = ImageEnhance.Contrast(img)
		out_img = enhancer.enhance(c_rate)
		flag = True
		en = "Contrast ({})".format(c_rate)

	elif enhance_type == 'Brightness':

		c_rate = st.sidebar.slider("Brightness",0.5,3.5, 1.2)
		enhancer = ImageEnhance.Brightness(img)
		out_img = enhancer.enhance(c_rate)
		flag = True
		en = "Brightness ({})".format(c_rate)

	elif enhance_type == 'Sharpness':
		s_rate = st.sidebar.slider("Sharpness",0.0,2.0, 1.2)
		enhancer = ImageEnhance.Sharpness(img)
		out_img = enhancer.enhance(s_rate)
		flag = True
		en = "Sharpness ({})".format(s_rate)

	elif enhance_type == 'Color':
		s_rate = st.sidebar.slider("Color", 0.0, 2.0, 0.8)
		enhancer = ImageEnhance.Color(img)
		out_img = enhancer.enhance(s_rate)
		flag = True
		en = "Color ({})".format(s_rate)

	
	elif enhance_type == 'Resize':
		img_w, img_h = img.size
		if st.sidebar.checkbox('Maintain Aspect Ratio'):
			ratio = st.sidebar.slider('Ratio', 0.05, 2.0, 1.0)
			w = int(img_w*ratio)
			h = int(img_h*ratio)
			out_img = img.resize((w,h))
		else:
			w = st.sidebar.slider('Width', 10, 1000, img_w)
			h = st.sidebar.slider('Height', 10, 1000, img_h)
			out_img = img.resize((w, h))

		en = "Resize ({}, {})".format(w, h)

		flag = True
	else:
		out_img = img
	
	if flag == True:
		st.text('Image After Enhancement {} - {}'.format(i, en))
		st.image(out_img)


	return out_img


def process_image(out_image):
	
	st.subheader("Detection")

	face_detector_mtcnn = MTCNN()
	model = FacialExpressionModel('model/model.json', 'model/model_weights.h5')
	frame = np.array(out_image)
	
	if len(frame.shape) == 3 :
			
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	else:

		gray = frame
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
		
	dict_faces = face_detector_mtcnn.detect_faces(frame_rgb) 
	faces =[]
	for dict_face in dict_faces:
		x,y,w,h = dict_face['box'][:]
		faces.append([x,y,w,h])

	st.text("Number of faces detected: {}".format(len(faces)))

	i = 1
	for (x,y,w,h) in faces:
		face = gray[y:y+h , x:x+h]
		roi = cv2.resize(face, (64,64))

		pred, confidence = model.predict_emotion(roi[np.newaxis, :, : , np.newaxis])

		cv2.putText(frame, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.rectangle(frame, (x,y), (x+w, y+h) , (255,255,0),2)
		st.text(str(i) + ". " + pred.capitalize())
		i+=1

	st.image(frame)

	if st.button('Save Detection'):
		plt.savefig(frame, 'image.png')


def main():
	
	st.title("Emotion Classifier")
	st.subheader("Feel the emotion, I'll tell you what it is.")

	activites = ["Detection", "About"]
	choice = st.sidebar.radio("Select Activity", activites)

	if choice == "Detection":
		# st.subheader("Emotion Classifier")
		upload_file = st.file_uploader("Upload an image", type=("png", "jpg", "jpeg"))

		img = 0
		if upload_file is not None:
			img = Image.open(upload_file)
			w , h = img.size
			nw = min(400,w)
			nh = int(nw*h/w)
			img = img.resize((nw,nh))
			st.text("Original Image")
			st.image(img)

		flag = True
		num = int(st.sidebar.number_input('Number of Enhancements',min_value = int(0),  max_value = int(5), value = int(0), format = '%d'))
		arr = list(['Brightness', 'Color', 'Contrast', 'Sharpness', 'Resize'])
		for i in range(int(num)):
			try:
				enhance_type = st.sidebar.selectbox("Enhancement {}".format(i+1), arr, i)
				img = enhancer(img, enhance_type, i)
			except:
				st.text('Please choose a different Enhancement.')
			# st.stop()
		
		if st.button('Save Enhanced Image'):
			plt.savefig(img, 'image.png')

		process = st.sidebar.button('Process Image')

		if process:				
			if upload_file is not None:
				process_image(img)
			else:
				st.text('Please upload a file.')

	else:
		st.write("This is an attempt to perform emotion recognition from facial expressions. There is added functionality to format the image as required. The model will run on the formatted image.")

		st.write("~ A project by Eklavya Jain")

if __name__ == "__main__":
	main()

