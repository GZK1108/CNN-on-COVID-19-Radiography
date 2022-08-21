import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import Adam,Adamax
from keras.layers import Input, Lambda, Dense, Flatten , Dropout , MaxPool2D
from keras.models import Model , Sequential
from keras.applications.nasnet import NASNetMobile

# Data Reading

import os

# Data Processing

import numpy as np
import pandas as pd
import cv2

# Data Analysis


import matplotlib.pyplot as plt

# Data Modeling & Model Evaluation

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Grad-CAM

levels = ['Normal', 'COVID']
path = "C:/Users/11453/PycharmProjects/riskassessment/data/COVID-19_Radiography_Dataset"
data_dir = os.path.join(path)

data = []
for id, level in enumerate(levels):
    for file in os.listdir(os.path.join(data_dir, level)):
        data.append(['{}/{}'.format(level, file), level])

data = pd.DataFrame(data, columns=['image_file', 'corona_result'])

data['path'] = path + '/' + data['image_file']
data['corona_result'] = data['corona_result'].map({'Normal': 'Negative', 'COVID': 'Positive'})

df = pd.DataFrame()
df['corona_result'] = ['Positive', 'Negative']
df['Count'] = [len(data[data['corona_result'] == 'Positive']), len(data[data['corona_result'] == 'Negative'])]
df = df.sort_values(by=['Count'], ascending=False)

all_data = []

# Storing images and their labels into a list for further Train Test split

for i in range(len(data)):
    image = cv2.imread(data['path'][i])
    image = cv2.resize(image, (70, 70)) / 255.0
    label = 1 if data['corona_result'][i] == "Positive" else 0
    all_data.append([image, label])

x = []
y = []

for image, label in all_data:
    x.append(image)
    y.append(label)

# Converting to Numpy Array
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print(x_train.shape, x_test.shape, x_val.shape, y_train.shape, y_test.shape, y_val.shape)

# 训练模型
batch_size = 32
epochs = 30

base_model = NASNetMobile(weights = 'imagenet', include_top=False, input_tensor=Input(shape=(70,70,3)))
headmodel = base_model.output
headmodel = AveragePooling2D(pool_size =(3, 3))(headmodel)
headmodel = Flatten(name ='Flatten')(headmodel)
headmodel = Dense(64, activation = 'relu')(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(2, activation = 'softmax')(headmodel)

model = Model(inputs = base_model.input, outputs = headmodel)

for layers in base_model.layers:
    layers.trainable = False


# opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam', metrics = ['accuracy'])


hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
print(score)
# print(hist.history['loss'])


model.save('nasnet.h5')

# 创建一个绘图窗口,画图
x = range(1, epochs + 1)
plt.figure()
acc = hist.history['accuracy']
loss = hist.history['loss']

plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, loss, 'y', label='Training loss')
plt.title('NasNet')
plt.xlabel('Epochs')
plt.ylabel('Acc and Loss')
plt.legend()

plt.savefig("nasnet.png")
plt.show()
