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
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

batch_size = 400
samples_per_epoch = 20000
nb_epoch = 10
img_height = 66
img_width = 200
img_channels=3
INPUT_SHAPE = (img_height, img_width, img_channels)

#data_dir='/home/vinayakpc/to_gpu/output_cam/'
#csv_path='/home/vinayakpc/to_gpu/output_csv/img_seg_ctrl.csv'
data_dir="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_cam\\"
csv_path="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_csv\\img_seg_ctrl.csv"

def preprocess(img):
    img = img[200:, :, :]
    img = cv2.resize(img, (img_width, img_height), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img

def NVIDIA_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    #model.summary()
    return model

model = NVIDIA_model()
model = load_model('my_aug_cnn')

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
predicted_control.throttle = 0.45
predicted_control.brake = 0.0
predicted_control.steer = 0.0
predicted_control.hand_brake = False
predicted_control.manual_gear_shift = False

def save_cam_nor(image):
    image=np.array(image.raw_data)
    image=np.reshape(image,(800,600,4))
    image=image[:,:,:3]
    image=preprocess(image)
    image=np.expand_dims(image, axis=0)
    result=model.predict(image)
    steering=result[0][0]
    predicted_control.steer = np.float32(steering).item()
    my_car.apply_control(predicted_control)

camera.listen(lambda image: save_cam_nor(image))
time.sleep(60)
predicted_control.throttle = 0.0
predicted_control.brake = 0.5
predicted_control.steer = 0.0
predicted_control.hand_brake = False
predicted_control.manual_gear_shift = False
my_car.apply_control(predicted_control)
camera.destroy()