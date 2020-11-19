import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
IMG_SIZE = 24

# Collect
train_idg = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)
valid_idg = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)

train_generator = train_idg.flow_from_directory(directory="./data/eyes/train", target_size=(IMG_SIZE,IMG_SIZE), color_mode="grayscale", batch_size=32, class_mode="binary", shuffle=True, seed=42)
valid_generator = valid_idg.flow_from_directory(directory="./data/eyes/valid", target_size=(IMG_SIZE,IMG_SIZE), color_mode="grayscale", batch_size=32, class_mode="binary", shuffle=True, seed=42)

# Train
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, epochs=20)

# Save model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

# Evaluate model
loss, acc = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)
print(acc)