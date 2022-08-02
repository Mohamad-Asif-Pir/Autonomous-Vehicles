from operator import mod
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import csv
import cv2
import random
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

batch_size = 100
#samples_per_epoch = 150000
nb_epoch = 4
img_height = 66
img_width = 200
img_channels=3
INPUT_SHAPE = (img_height, img_width, img_channels)

inputs = keras.Input(shape=(66,200,3))

def conv_block1(input,a,b,c):
    conv2dx1=keras.layers.Conv2D(a,b,c)
    act1=keras.layers.ReLU()
    tmp_x=conv2dx1(input)
    tmp_x=act1(tmp_x)
    return tmp_x

def fc_block1(input,a):
    dense1=keras.layers.Dense(a)
    tmp_x=dense1(input)
    if(a!=1):
        act1=keras.layers.ReLU()
        tmp_x=act1(tmp_x)
    return tmp_x

lambda1=keras.layers.Lambda(lambda x: x/127.5-1.0)
x=lambda1(inputs)
x=conv_block1(x,24,(5,5),(2,2))
x=conv_block1(x,36,(5,5),(2,2))
x=conv_block1(x,48,(5,5),(2,2))
x=conv_block1(x,64,(3,3),(1,1))
x=conv_block1(x,64,(3,3),(1,1))
drp1=keras.layers.Dropout(0.5)
x=drp1(x)
flat1=keras.layers.Flatten()
x=flat1(x)

branches=[]
branch1=fc_block1(x,100)
branch1=fc_block1(branch1,50)
branch1=fc_block1(branch1,10)
branch1=fc_block1(branch1,1)
branches.append(keras.Model(inputs=inputs, outputs=branch1, name="branch1"))

branch2=fc_block1(x,100)
branch2=fc_block1(branch2,50)
branch2=fc_block1(branch2,10)
branch2=fc_block1(branch2,1)
branches.append(keras.Model(inputs=inputs, outputs=branch2, name="branch2"))

branch3=fc_block1(x,100)
branch3=fc_block1(branch3,50)
branch3=fc_block1(branch3,10)
branch3=fc_block1(branch3,1)
branches.append(keras.Model(inputs=inputs, outputs=branch3, name="branch3"))

branch4=fc_block1(x,100)
branch4=fc_block1(branch4,50)
branch4=fc_block1(branch4,10)
branch4=fc_block1(branch4,1)
branches.append(keras.Model(inputs=inputs, outputs=branch4, name="branch4"))

#data_dir="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_cam\\"
#csv_path="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_csv\\img_seg_ctrl.csv"
#data_dir='/home/vinayakpc/out/output_cam/'
#csv_path='/home/vinayakpc/out/output_csv/img_seg_ctrl.csv'
data_dir='//home//choyya//out1//output_cam//'
csv_path='//home//choyya//out1//output_csv//img.csv'
#import subprocess
#import os

#subprocess.call(r'net use \\\\10.100.12.113\c$\home\choyya\out1\output_cam\ /user:choyya password:Choyya@321', shell=True) #connect with network drive

#print('Connection Established')

#data_dir = ('\\\\10.100.12.113\\c$\\home\\choyya\\out1\\output_csv\\img.csv')  #display the name of the files in directory

#print(x)

from sklearn.model_selection import train_test_split

def load_data(direction):
    X_path = []
    Y=[]
    img_name_list=os.listdir(data_dir)
    with open(csv_path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            #t1=row[0][2:].rjust(6, '0')
            #row[0]=row[0][:2]+t1
            if(row[0] in img_name_list and row[3]==str(direction)):
                X_path.append(row[0])
                if(row[0][6]=='c'):
                    Y.append(float(row[2]))
                elif (row[0][6]=='l'):
                    Y.append(float(row[2])+0.35)
                else:
                    Y.append(float(row[2])-0.35)
    print(len(X_path))
    X_train, X_valid, y_train, y_valid = train_test_split(X_path, Y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid

def preprocess(img):
    img = img[200:, :, :]
    img = cv2.resize(img, (img_width, img_height), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img

def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_shift(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    bright_factor = 0.3
    x = random.randint(0, image.shape[1])
    y = random.randint(0, image.shape[0])
    width = random.randint(image.shape[1], image.shape[1])
    if(x + width > image.shape[1]):
        x = image.shape[1] - x
    height = random.randint(image.shape[0], image.shape[0])
    if(y + height > image.shape[0]):
        y = image.shape[0] - y
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*bright_factor
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augument(image, steering_angle, range_x=100, range_y=10,random_flip_1=True):
    if(random_flip_1):
        image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_shift(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def batcher(image_paths, steering_angles, batch_size, training_flag,random_flip_1=True):
    images = np.empty([batch_size, img_height, img_width, img_channels])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(len(image_paths)):
            img_name = image_paths[index]
            steering_angle = steering_angles[index]
            img=cv2.imread(data_dir+img_name)
            if training_flag and np.random.rand() < 0.6:
                img, steering_angle = augument(img, steering_angle,random_flip_1)
            images[i] = preprocess(img)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

data=[]
data_1=load_data(1)
data_2=load_data(2)
data_3=load_data(3)
data_4=load_data(4)

def train_model(model, X_train, X_valid, y_train, y_valid,random_flip_1=True):
    checkpoint = ModelCheckpoint("/home/choyya/chkpt/"+'model-{val_loss:03f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    model.compile(loss='mse', optimizer=Adam(learning_rate=1.0e-4))
    model.fit(batcher(X_train, y_train, batch_size, True,random_flip_1),
                        epochs=nb_epoch,
                        max_queue_size=1,
                        steps_per_epoch=samples_per_epoch,
                        validation_data=batcher(X_valid, y_valid, batch_size, False),
                        validation_steps=len(X_valid)/batch_size,
                        callbacks=[checkpoint],
                        verbose=1)

# model=branches[3]
# model.summary()
# for i in model.layers:
#     print(i)


model=branches[3]
train_model(model,*data_4,random_flip_1=True)
model.save('my_branch_4')

for tmp in model.layers[:-7]:
    tmp.trainable = False

model=branches[0]
train_model(model,*data_1,random_flip_1=False)
model.save('my_branch_1')

model=branches[1]
train_model(model,*data_2,random_flip_1=True)
model.save('my_branch_2')

model=branches[2]
train_model(model,*data_3,random_flip_1=False)
model.save('my_branch_3')
