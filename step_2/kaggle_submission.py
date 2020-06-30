from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19 
from keras.applications.resnet50 import ResNet50
from keras.applications import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import backend as K
import tensorflow as tf
import time
import os
import gc
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import fbeta_score
import cv2
from tqdm import tqdm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History
import pandas as pd
import numpy as np
import re
import os

test = os.listdir('test-jpg')
test_add = os.listdir('test-jpg-additional')

DIR = 'test-jpg/test_'
DIR2 = 'test-jpg-additional/file_'
test_images = []

for i in range(0, len(test)):
      test_images.append(DIR + str(i) + '.jpg')

for i in range(0, len(test_add)):
      test_images.append(DIR2 + str(i) + '.jpg')

labels = ['haze', 'road', 'blow_down', 'primary', 'clear', 'slash_burn', 'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'cultivation', 'water', 'artisinal_mine', 'blooming', 'agriculture', 'selective_logging', 'habitation']
labels.sort()

print(labels)

from keras.models import load_model
import gc 

input_size = 128
input_channels = 3
bs=128

img_list = []

for f in test_images:
    img_test = cv2.imread(f)  
    img_list.append(cv2.resize(img_test, (input_size, input_size))) 

img_list = np.array(img_list, np.float16) / 255.


def prediction_model(use_model, bs, imgs):  
    if use_model == 'inceptionv3':
        base_model = InceptionV3(include_top=False, input_shape=(input_size, input_size, input_channels))
    elif use_model == 'resnet50':
        base_model = ResNet50(include_top=False, input_shape=(input_size, input_size, input_channels))
    elif use_model == 'vgg16':
        base_model = VGG16(include_top=False, input_shape=(input_size, input_size, input_channels))
    elif use_model == 'mobilenet':
        base_model = MobileNet(include_top=False, input_shape=(input_size, input_size, input_channels))
    elif use_model == 'mobilenetv2':
        base_model = MobileNetV2(include_top=False, input_shape=(input_size, input_size, input_channels))
    elif use_model == 'vgg19':
        base_model = VGG19(include_top=False, input_shape=(input_size, input_size, input_channels))

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(17, activation='sigmoid'))

    if use_model == 'inceptionv3':
        model.load_weights('inceptionv3.h5')
    elif use_model == 'resnet50':
        model.load_weights('resnet50.h5')
    elif use_model == 'vgg16':
        model.load_weights('vgg16.h5')
    elif use_model == 'mobilenet':
        model.load_weights('mobilenet.h5')
    elif use_model == 'mobilenetv2':
        model.load_weights('mobilenetv2.h5')
    elif use_model == 'vgg19':
        model.load_weights('vgg19.h5')

    resp = model.predict(np.array(imgs), batch_size = bs)
    resp_list = []


    for img in resp:
        for label in list(img):
            resp_list.append(1 if label>0.2 else 0)

    return resp_list


def return_filename(x):
    result = re.search('/(.*).jpg', x)
    return result.group(1)

def ensemble(resp_array, min_votes):
    img_resp = []
    final_resp = []

    i = 0
    for label_images in np.nditer(resp_array):
        if label_images<min_votes:
            img_resp.append(False)
        else:
            img_resp.append(True)
    
        i = i+1

        if i%17 == 0:
            final_resp.append(img_resp)
            img_resp = []
    
    resp_kaggle = np.array(final_resp)

    tags_resp = [] 

    labels_array = np.array(labels)

    for r in resp_kaggle:
        tags_resp.append(" ".join(labels_array[np.where(r)[0]]))

    img_idx = [return_filename(x) for x in test_images ]

    df_kaggle = pd.DataFrame({'image_name':img_idx, 'tags':tags_resp})

    return df_kaggle


start = time.time()

'Comment the networks that are unnecessary because of the time counting'

#resp_inception_list = prediction_model('inceptionv3', bs, img_list)
#print('INCEPTION OK\n\n')
#resp_resnet_list = prediction_model('resnet50', bs, img_list)
#print('RESNET OK\n\n')
#resp_vgg_list = prediction_model('vgg16', bs, img_list)
#print('VGG OK\n\n')
#resp_mobilenet_list = prediction_model('mobilenet', bs, img_list)
#print('MOBILENET OK\n\n')
#resp_mobilenetv2_list = prediction_model('mobilenetv2', bs, img_list)
#print('MOBILENETV2 OK\n\n')
resp_vgg19_list = prediction_model('vgg19', bs, img_list)
print('VGG19 OK\n\n') 

'sum here all the networks that you want to do the ensemble'
resp_sum = np.array(resp_vgg19_list)

'put here how much votes are necessary'
df_kaggle_final = ensemble(resp_sum, 1)

end = time.time()

print('Time: ' + str(end-start))
df_kaggle_final.to_csv('vgg19-test-submission.csv', index=False)

