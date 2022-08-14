import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.applications.resnet import ResNet50
import tensorflow as tf
from keras.layers import  Flatten, Dense, Dropout
from keras.models import Sequential


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

print('Number of Duplicated Samples: %d' % (data.duplicated().sum()))
print('Number of Total Samples: %d' % (data.isnull().value_counts()))

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

fig.show()

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
plt.imshow(image)
plt.axis('off')


plt.title('B channel', fontsize=14)
plt.imshow(image[:, :, 0])
plt.axis('off')


all_covid = []
all_normal = []

all_normal.extend(glob(os.path.join(path, "Normal/*.png")))
all_covid.extend(glob(os.path.join(path, "COVID/*.png")))

random.shuffle(all_normal)
random.shuffle(all_covid)

images = all_normal[:50] + all_covid[:50]

fig = plt.figure(figsize=(18, 7))
fig.suptitle("Ben Grahamns Method of Analysis", fontsize=15)
columns = 4
rows = 2

for i in range(1, columns * rows + 1):
    img = cv2.imread(images[i])
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 512 / 10), -4, 128)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis(False)


def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(15, 8), nrows=2, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize=18)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()


chosen_image = cv2.imread(
    "C:/Users/11453/PycharmProjects/riskassessment/data/COVID-19_Radiography_Dataset/COVID/COVID-1002.png")

albumentation_list = [A.RandomFog(p=1), A.RandomBrightness(p=1),
                      A.RandomCrop(p=1, height=199, width=199), A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit=0.5, p=1)]

img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image=chosen_image)['image']
    img_matrix_list.append(img)

img_matrix_list.insert(0, chosen_image)

titles_list = ["Original", "RandomFog", "RandomBrightness", "RandomCrop", "Rotate", "RGBShift", "VerticalFlip",
               "RandomContrast"]

plot_multiple_img(img_matrix_list, titles_list, ncols=4, main_title="Different Types of Augmentations")

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


plt.figure(figsize=(20, 8))
sns.set(style="ticks", font_scale=1)
ax = sns.scatterplot(data=imageEDA, x="mean", y=imageEDA['stedev'], hue='corona_result', alpha=0.8);
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xticks(rotation=0, fontsize=12)
ax.set_xlabel('\nImage Channel Colour Mean', fontsize=14)
ax.set_ylabel('Image Channel Colour Standard Deviation', fontsize=14)
plt.title('Mean and Standard Deviation of Image Samples', fontsize=16)

plt.figure(figsize=(10, 8))
g = sns.FacetGrid(imageEDA, col="corona_result", height=5)
g.map_dataframe(sns.scatterplot, x='mean', y='stedev')
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=12)
g.fig.subplots_adjust(top=.7)
g.fig.suptitle('Mean and Standard Deviation of Image Samples', fontsize=15)
axes = g.axes.flatten()
axes[0].set_ylabel('Standard Deviation')
for ax in axes:
    ax.set_xlabel('\nMean')
g.fig.tight_layout()
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
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(70,70,3),pooling='avg')
restnet.summary()

model = Sequential()
model.add(restnet)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
model.fit(x_train,y_train, batch_size = batch_size, epochs=15)

score = model.evaluate(x_test,y_test)

model.save('model.h5')