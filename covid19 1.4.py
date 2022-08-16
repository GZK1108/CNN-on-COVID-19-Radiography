import warnings

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


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
import albumentations as A

# Data Analysis

import plotly.express as px
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
samples = 13808

data.head()

df = pd.DataFrame()
df['corona_result'] = ['Positive', 'Negative']
df['Count'] = [len(data[data['corona_result'] == 'Positive']), len(data[data['corona_result'] == 'Negative'])]
df = df.sort_values(by=['Count'], ascending=False)

fig = px.bar(df, x='corona_result', y='Count',
             color="corona_result", text_auto='', width=600,
             color_discrete_sequence=["orange", "purple"],
             template='plotly_dark')

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)


data['image'] = data['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 75))))

data.head()

n_samples = 3

fig, m_axs = plt.subplots(2, n_samples, figsize=(6 * n_samples, 3 * 4))

for n_axs, (type_name, type_rows) in zip(m_axs, data.sort_values(['corona_result']).groupby('corona_result')):
    n_axs[1].set_title(type_name, fontsize=15)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        picture = c_row['path']
        image = cv2.imread(picture)
        c_ax.imshow(image)
        c_ax.axis('off')

plt.figure()
image = cv2.imread(
    "C:/Users/11453/PycharmProjects/riskassessment/data/COVID-19_Radiography_Dataset/COVID/COVID-1002.png")

plt.axis('off')



all_covid = []
all_normal = []

all_normal.extend(glob(os.path.join(path, "Normal/*.png")))
all_covid.extend(glob(os.path.join(path, "COVID/*.png")))

random.shuffle(all_normal)
random.shuffle(all_covid)

images = all_normal[:50] + all_covid[:50]


columns = 4
rows = 2


albumentation_list = [A.RandomFog(p=1), A.RandomBrightness(p=1),
                      A.RandomCrop(p=1, height=199, width=199), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit=0.5, p=1)]

img_matrix_list = []
bboxes_list = []


titles_list = ["Original", "RandomFog", "RandomBrightness", "RandomCrop", "Rotate", "RGBShift", "VerticalFlip",
               "RandomContrast"]


mean_val = []
std_dev_val = []
max_val = []
min_val = []

for i in range(0, samples):
    mean_val.append(data['image'][i].mean())
    std_dev_val.append(np.std(data['image'][i]))
    max_val.append(data['image'][i].max())
    min_val.append(data['image'][i].min())

imageEDA = data.loc[:, ['image', 'corona_result', 'path']]
imageEDA['mean'] = mean_val
imageEDA['stedev'] = std_dev_val
imageEDA['max'] = max_val
imageEDA['min'] = min_val

imageEDA['subt_mean'] = imageEDA['mean'].mean() - imageEDA['mean']


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


model = load_model("model.h5")
score = model.evaluate(x_test, y_test)
print(score)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
cm = confusion_matrix(y_test, pred)
print(cm)

plt.figure(figsize=(16, 9), dpi=80)
ax = sns.heatmap(data=cm, annot=True, fmt='d', annot_kws={"fontsize":20}, cbar=False)
ax.set_title('Confusion matrix')  # 图标题
ax.set_xlabel('Predict',fontsize=34)  # x轴标题
ax.set_ylabel('True',fontsize=34)
# plt.xticks(fontsize=50)
# plt.yticks(fontsize=50)
plt.show()

print("准确度为：")
print(accuracy_score(y_test, pred, normalize=True, sample_weight=None))
print("精确度为:")
print(precision_score(y_test, pred, average='binary'))  # 测试集精确率
print("召回率为:")
print(recall_score(y_test, pred, average="binary"))

