from gtts import gTTS
import random
import playsound
import os
from PIL import Image
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
from keras.models import model_from_json

def speak(string):
    tts = gTTS(text=string, lang='en')
    rand = random.randint(1,20000000)
    audio_file = 'audio' + str(rand) + '.mp3'
    tts.save(audio_file)
    playsound.playsound(audio_file)
    os.remove(audio_file)

def load_model():
	json_file = open('model.json', 'r')
	loaded_model = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model)
	model.load_weights('model.h5')
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def predict(img, model):
    img = Image.fromarray(img, 'RGB').convert('L')
    img = imresize(img, (24,24)).astype('float32')
    img /= 255
    img = img.reshape(1, 24, 24, 1)
    prediction = model.predict(img)
    if prediction < 0.1:
	    prediction = 'Closed'
    elif prediction > 0.9:
	    prediction = 'Open'
    else:
	    prediction = 'Not Sure'
    
    return prediction