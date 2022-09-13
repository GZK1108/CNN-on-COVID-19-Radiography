import warnings

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.models import load_model

# Data Reading

import os
from glob import glob
from PIL import Image

# Data Processing

import numpy as np
import pandas as pd
import cv2
import random


# Data Analysis


import matplotlib.pyplot as plt
import seaborn as sns

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
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = load_model("C:/Users/11453/PycharmProjects/riskassessment/nasnet1.h5")
score = model.evaluate(x_test, y_test)
# score = model.evaluate(x_train, y_train)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
cm = confusion_matrix(y_test, pred)


plt.figure(figsize=(12, 9), dpi=80)
x_ticks =['Negative','Positive']
y_ticks =['Negative','Positive']
ax = sns.heatmap(data=cm, xticklabels=x_ticks, yticklabels=y_ticks,annot=True, fmt='d', annot_kws={"fontsize":20}, cmap='Blues')
ax.set_title('NASNer',fontsize=20)  # 图标题
ax.set_xlabel('Predict',fontsize=15)  # x轴标题
ax.set_ylabel('True',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('confusion matrix nasnet.png')

print("accuracy:",round(accuracy_score(y_test, pred, normalize=True, sample_weight=None),3))
print("precision:",round(precision_score(y_test, pred, average='binary'),3))  # 测试集精确率
print("recall:",round(recall_score(y_test, pred, average="binary"),3))
print("F1 score:",round(f1_score(y_test, pred, average="binary"),3))
plt.show()

"""# 训练集
pred1 = model.predict(x_train)
pred1 = np.argmax(pred1, axis=1)

print("accuracy:",round(accuracy_score(y_train, pred1, normalize=True, sample_weight=None),3))
print("precision:",round(precision_score(y_train, pred1, average='binary'),3))  # 测试集精确率
print("recall:",round(recall_score(y_train, pred1, average="binary"),3))
print("F1 score:",round(f1_score(y_train, pred1, average="binary"),3))"""




