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
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

print(x_train.shape, x_test.shape, x_val.shape, y_train.shape, y_test.shape, y_val.shape)


model = load_model("C:/Users/11453/PycharmProjects/riskassessment/resnetCOVID19.h5")
# score = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
cm = confusion_matrix(y_test, pred)


plt.figure(figsize=(12, 9), dpi=60)
x_ticks =['Negitive','Positive']
y_ticks =['Negitive','Positive']
ax = sns.heatmap(data=cm, xticklabels=x_ticks,yticklabels=y_ticks,annot=True, fmt='d', annot_kws={"fontsize":20}, cmap='Blues')
ax.set_title('Confusion matrix',fontsize=20)  # 图标题
ax.set_xlabel('Predict',fontsize=15)  # x轴标题
ax.set_ylabel('True',fontsize=15)
# plt.xticks(fontsize=50)
# plt.yticks(fontsize=50)

# plt.savefig('confusion matrix.png')

print("accuracy:",round(accuracy_score(y_test, pred, normalize=True, sample_weight=None),3))
print("precision:",round(precision_score(y_test, pred, average='binary'),3))  # 测试集精确率
print("recall:",round(recall_score(y_test, pred, average="binary"),3))
print("F1 score:",round(f1_score(y_test, pred, average="binary"),3))
plt.show()
"""# 测试集准确率
plt.figure()
accuracy = accuracy_score(y_test, pred, normalize=True)
plt.plot(accuracy, label='testing accuracy')
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
# plt.savefig("./result/每一轮准确度图片.png")
plt.legend()
plt.show()

plt.plot(loss, label='testing loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig("./result/每一轮损失值图片.png")
plt.legend()
plt.show()"""



