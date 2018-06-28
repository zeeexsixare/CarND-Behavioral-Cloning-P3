"""
CarND-Behavioral-Cloning-P3 Project
Philip Lee 6/27/18

Udacity CarND: ami-c4c4e3a4

AWS CHECKLIST
1. CREATE AWS INSTANCE
2. LOG INTO INSTANCE (carnd, carnd)
3. PIP INSTALL OPENCV-PYTHON
4. PIP INSTALL TENSORFLOW
5. PIP INSTALL KERAS==1.2.1
5. ZIP EVERYTHING UP AND UPLOAD INTO AWS INSTANCE (scp data.zip carnd@X.X.X.X:/home/carnd/CarND-Behavioral-Cloning-P3/)
6. RUN CLONE.PY
7. DOWNLOAD MODEL BACK FROM AWS INSTANCE (scp carnd@X.X.X.X:/home/carnd/CarND-Behavioral-Cloning-P3/model.h5 .)
8. RUN PYTHON DRIVE.PY MODEL.H5
9. DON'T CRASH INTO THINGS
"""


import csv
import cv2
import numpy as np
import tensorflow as tf

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

correction = 0.1
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('\\')[-1]
		current_path = './data/IMG/' + filename

		#INTERPRET AND ADD REGULAR ORIENTATION IMAGE
		image = cv2.imread(current_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(image)
		#INTERPRET AND ADD FLIPPED ORIENTATION IMAGE
		image_flipped = np.fliplr(image)
		images.append(image_flipped)

	#ADD CENTER MEASUREMENTS
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(-measurement)#FLIPPED MEASUREMENTS
	
	measurements.append(measurement+correction)#LEFT IMAGE WITH 2.5 DEGREE OFFSET
	measurements.append(-(measurement+correction))#FLIPPED LEFT IMAGE WITH 2.5 DEGREE OFFSET
	
	measurements.append(measurement-correction)#RIGHT IMAGE WITH 2.5 DEGREE OFFSET
	measurements.append(-(measurement-correction))#FLIPPED RIGHT IMAGE WITH 2.5 DEGREE OFFSET

	
	
	
#print(filename)	
	
X_train =  np.array(images)
y_train =  np.array(measurements)
#print(X_train[0])
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.normalization import BatchNormalization

"""OLD MODEL
model = Sequential()
model.add(Lambda(lambda X_train: (X_train / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
"""

#NVIDIA MODEL
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(BatchNormalization())
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, verbose=1, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')


