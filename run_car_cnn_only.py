import time
import carla
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D # to add pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers
from keras.layers import Dropout
from keras.models import load_model
def convolutional_model():
    # create model
    model = Sequential()
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['accuracy'])
    return model

# build the model
model = convolutional_model()
model = load_model('model')

import cv2

client = carla.Client('10.100.12.113', 2000)
client.set_timeout(10.0)
world = client.get_world()
map=world.get_map()
blueprint_lib=world.get_blueprint_library()

my_car = world.get_actors().filter('vehicle.*')[0]

camera_bp=blueprint_lib.find("sensor.camera.rgb")
camera_bp.set_attribute('sensor_tick', '0.1')

relative_transform=carla.Transform(carla.Location(x=1.6,y=0,z=1.5))
camera = world.spawn_actor(camera_bp, relative_transform, attach_to=my_car)

predicted_control = carla.VehicleControl()
predicted_control.throttle = 0.3
predicted_control.brake = 0.0
predicted_control.steer = 0.0
predicted_control.hand_brake = False
predicted_control.manual_gear_shift = False

def save_cam_nor(image):
    image=np.array(image.raw_data)
    image=np.reshape(image,(800,600,4))
    image=image[:,:,:3]
    #image=cv2.resize(image,dsize=(600,600),interpolation = cv2.INTER_NEAREST)
    image=(np.array(image.data)/127.5)-1
    image=np.expand_dims(image, axis=0)
    result=model.predict(image)
    #accel=result[0][0]
    steering=result[0][0]
    #predicted_control.throttle = np.float32(accel).item()*1.2
    gt1= np.float32(steering).item()/4
    if(gt1>0.65):
        gt1=0.65
    predicted_control.steer = gt1
    my_car.apply_control(predicted_control)

camera.listen(lambda image: save_cam_nor(image))
time.sleep(600)
predicted_control.throttle = 0.0
predicted_control.brake = 0.5
predicted_control.steer = 0.0
predicted_control.hand_brake = False
predicted_control.manual_gear_shift = False
my_car.apply_control(predicted_control)
camera.destroy()