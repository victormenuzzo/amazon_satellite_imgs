# -*- coding: utf-8 -*-
"""train_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XkC9UXJ2X1l4AYa5ctY8rmbC5oB-r97a
"""

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
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

'''
Descomentar essa linha da primeira vez que for usar para poder dezipar a pasta de imagens
'''
#!apt-get install tar
!apt-get install p7zip-full

# Login
from google.colab import drive
drive.mount('/content/drive')

classifications = !unzip -q "/content/drive/My Drive/Colab Notebooks/TCC/train_v2.csv.zip"
classifications = pd.read_csv('train_v2.csv', sep=',')

!7z e '/content/drive/My Drive/train-jpg.tar.7z'
train = !tar -xvf 'train-jpg.tar'

'''
Passa o caminho de todas as imagens na ordem correta para train_images
'''
DIR = 'train-jpg/train_'
train_images = []
for i in range(0, len(train)):
  train_images.append(DIR + str(i) + '.jpg')

"""## Callbacks"""

class TimeCallback(tf.keras.callbacks.Callback):
    global inicio_treino, inicio_teste

    def on_train_begin(self, batch, logs=None):
        print('inicio do treino')
        self.inicio_treino = time.time()

    def on_train_end(self, batch, logs=None):
        print('fim do treino')
        print('durou: '+str(time.time() - self.inicio_treino))
        train_time.append(time.time() - self.inicio_treino)
    
    def on_test_begin(self, batch, logs=None):
        print('inicio do teste')
        self.inicio_teste = time.time()

    def on_test_end(self, batch, logs=None):
        #print('fim do teste')
        print('durou: '+str(time.time() - self.inicio_teste))
        test_time.append(time.time() - self.inicio_teste)

"""## Leitura das Imagens"""

input_size = 128
input_channels = 3
x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('train_v2.csv', sep=',')


flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (input_size, input_size)))
    y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)
split = 35000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

"""## Modelo"""

labels

train_time = []
test_time = []

#base_model = VGG16(include_top=False, input_shape=(input_size, input_size, input_channels))
#base_model = ResNet50(include_top=False, input_shape=(input_size, input_size, input_channels))
#base_model = MobileNet(include_top=False, input_shape=(input_size, input_size, input_channels))
base_model = MobileNetV2(include_top=False, input_shape=(input_size, input_size, input_channels))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(17, activation='sigmoid'))
print(model.summary()) 

history = History()
callbacks = [history,
            EarlyStopping(monitor="val_loss", patience=5, verbose=1, min_delta=1e-4),
            TimeCallback()]
bs = 128

model.compile(optimizer = Adam(lr=1e-4), loss = "binary_crossentropy", metrics = ["accuracy", "binary_accuracy"])
history = model.fit(x=np.array(x_train), y=np.array(y_train), epochs=5, batch_size = bs, verbose = 1, validation_data = (np.array(x_valid), np.array(y_valid)), callbacks = callbacks)

p_valid = model.predict(np.array(x_valid), batch_size = bs)
p_valid = model.predict(x_valid, batch_size=128)
model.save_weights('mobilev2-test2.h5')
print(100*'-')
print('Y_VALID')
print(y_valid)
print(100*'-')
print('P_VALID')
print(p_valid)
print(100*'-')
print('F_VALID')
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
respostas = pd.DataFrame(np.array(p_valid) > 0.2, columns=labels).astype(int)
gabarito = pd.DataFrame(y_valid, columns=labels)

csv_resp = []
for labels_resp in labels:  
  #print(labels_resp)
  teste = gabarito[labels_resp] == 1
  #print(sum(teste))
  a = respostas[teste]
  b = gabarito[teste]
  c = b[labels_resp] == a[labels_resp]
  #print(sum(c))
  csv_resp.append([labels_resp, sum(teste), sum(c), round(sum(c)/sum(teste),2)*100])

csv_resp = pd.DataFrame(csv_resp)
csv_resp.to_csv('mobilev2_resp.csv')

train_time

np.mean(test_time)