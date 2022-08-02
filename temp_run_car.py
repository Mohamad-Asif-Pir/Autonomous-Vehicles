import time
import carla
import numpy as np

from tensorflow import keras
from keras.models import load_model
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

dropout_vec = [0.99] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 0.99] * 5
dropout_vec_cntr=0
batch_size = 100
samples_per_epoch = 2000
nb_epoch = 10
img_height = 66
img_width = 200
img_channels=3
INPUT_SHAPE = (img_height, img_width, img_channels)

inputs = keras.Input(shape=(66,200,3))

def conv_block1(input,a,b,c):
    global dropout_vec_cntr
    conv2dx1=keras.layers.Conv2D(a,b,c,padding="valid")
    bn1=keras.layers.BatchNormalization()
    drp1=keras.layers.Dropout(dropout_vec[dropout_vec_cntr])
    dropout_vec_cntr+=1
    act1=keras.layers.ReLU()
    tmp_x=conv2dx1(input)
    tmp_x=bn1(tmp_x)
    tmp_x=drp1(tmp_x)
    tmp_x=act1(tmp_x)
    return tmp_x

def fc_block1(input,a):
    global dropout_vec_cntr
    dense1=keras.layers.Dense(a)
    drp1=keras.layers.Dropout(dropout_vec[dropout_vec_cntr])
    dropout_vec_cntr+=1
    act1=keras.layers.ReLU()
    tmp_x=dense1(input)
    tmp_x=drp1(tmp_x)
    tmp_x=act1(tmp_x)
    return tmp_x

x=conv_block1(inputs,32,5,2)
x=conv_block1(x,32,3,1)
x=conv_block1(x,64,3,2)
x=conv_block1(x,64,3,1)
x=conv_block1(x,128,3,2)
x=conv_block1(x,128,3,1)
x=conv_block1(x,256,3,1)
#x=conv_block1(x,256,3,1)
reshape1=keras.layers.Reshape([-1,np.prod(x.get_shape()[1:])])
x=reshape1(x)
x=fc_block1(x,512)
x=fc_block1(x,512)
x=fc_block1(x,512)
branches=[]
branch1=fc_block1(x,256)
branch1=fc_block1(branch1,256)
branch1=fc_block1(branch1,1)
branches.append(keras.Model(inputs=inputs, outputs=branch1, name="branch1"))

branch2=fc_block1(x,256)
branch2=fc_block1(branch2,256)
branch2=fc_block1(branch2,1)
branches.append(keras.Model(inputs=inputs, outputs=branch2, name="branch2"))

branch3=fc_block1(x,256)
branch3=fc_block1(branch3,256)
branch3=fc_block1(branch3,1)
branches.append(keras.Model(inputs=inputs, outputs=branch3, name="branch3"))

branch4=fc_block1(x,256)
branch4=fc_block1(branch4,256)
branch4=fc_block1(branch4,1)
branches.append(keras.Model(inputs=inputs, outputs=branch4, name="branch4"))

def preprocess(img):
    img = img[200:, :, :]
    img = cv2.resize(img, (img_width, img_height), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img

model_1 = branches[0]
model_1 = load_model(r'C:\Users\Lenovo\Downloads\images\Exp_4\my_branch_1')
model_2 = branches[1]
model_2 = load_model(r'C:\Users\Lenovo\Downloads\images\Exp_4\my_branch_2')
model_3 = branches[2]
model_3 = load_model(r'C:\Users\Lenovo\Downloads\images\Exp_4\my_branch_3')
model_4 = branches[3]
model_4 = load_model(r'C:\Users\Lenovo\Downloads\images\Exp_4\my_branch_4')

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
    direction=4
    if(map.get_waypoint(my_car.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving).is_junction): 
        lights=my_car.get_light_state()
        if(lights==carla.VehicleLightState.LeftBlinker):
            direction=1
        elif(lights==carla.VehicleLightState.RightBlinker):
            direction=3
        else:
            direction=2
    if(direction==1):
        result=model_1.predict(image)
    elif(direction==2):
        result=model_2.predict(image)
    elif(direction==3):
        result=model_3.predict(image)
    else:
        result=model_4.predict(image)
    steering=result[0][0]
    predicted_control.steer = np.float32(steering).item()
    predicted_control.throttle = 0.45
    my_car.apply_control(predicted_control)

print("start driving")
camera.listen(lambda image: save_cam_nor(image))
time.sleep(60)
print("done driving")
predicted_control.throttle = 0.0
predicted_control.brake = 0.5
predicted_control.steer = 0.0
predicted_control.hand_brake = False
predicted_control.manual_gear_shift = False
my_car.apply_control(predicted_control)
camera.destroy()