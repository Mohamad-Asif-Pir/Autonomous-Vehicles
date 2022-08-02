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
os.environ["CUDA_VISIBLE_DEVICES"]="2"

dropout_vec = [0.99] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 0.99] * 5
dropout_vec_cntr=0
batch_size = 100
samples_per_epoch = 200
nb_epoch = 2
img_height = 88
img_width = 200
img_channels=3
INPUT_SHAPE = (img_height, img_width, img_channels)

inputs = keras.Input(shape=(88,200,3))

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
x=conv_block1(x,256,3,1)
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

#data_dir="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_cam\\"
#csv_path="C:\\Users\\Vinayak\\Downloads\\pt\\CARLA_Latest\\WindowsNoEditor\\PythonAPI\\examples\\out\\output_csv\\img_seg_ctrl.csv"
data_dir='/home/choyya/out1/output_cam/'
csv_path='/home/choyya/out1/output_csv/img.csv'

from sklearn.model_selection import train_test_split

def flip1(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

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
                X_path.append(row[0]+'.png')
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

def augument(image, steering_angle, range_x=100, range_y=10):
    #image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_shift(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def batcher(image_paths, steering_angles, batch_size, training_flag):
    images = np.empty([batch_size, img_height, img_width, img_channels])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(len(image_paths)):
            img_name = image_paths[index]
            steering_angle = steering_angles[index]
            img=cv2.imread(data_dir+img_name)
            if training_flag and np.random.rand() < 0.6:
                img, steering_angle = augument(img, steering_angle)
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

def train_model(model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint("/home/choyya/chkpt/"+'model-{val_loss:03f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    model.compile(loss='mse', optimizer=Adam(learning_rate=1.0e-4))
    model.fit(batcher(X_train, y_train, batch_size, True),
                        epochs=nb_epoch,
                        max_queue_size=1,
                        steps_per_epoch=samples_per_epoch,
                        validation_data=batcher(X_valid, y_valid, batch_size, False),
                        validation_steps=len(X_valid)/batch_size,
                        callbacks=[checkpoint],
                        verbose=1)

model=branches[3]
train_model(model,*data_4)
model.save('my_branch_4')

for tmp in model.layers[:-9]:
    tmp.trainable = False

model=branches[0]
train_model(model,*data_1)
model.save('my_branch_1')

model=branches[1]
train_model(model,*data_2)
model.save('my_branch_2')

model=branches[2]
train_model(model,*data_3)
model.save('my_branch_3')


