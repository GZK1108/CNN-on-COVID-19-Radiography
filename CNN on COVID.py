# 在这里面写代码，模块化
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
from keras.preprocessing import image
from keras import layers, models
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, \
    accuracy_score, precision_score, f1_score

# Grad-CAM

import keras
import matplotlib.cm as cm

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
df = df.sort_values(by = ['Count'], ascending = False)

fig = px.bar(df, x = 'corona_result', y = 'Count',
             color = "corona_result", text_auto='', width = 600,
             color_discrete_sequence = ["orange", "purple"],
             template = 'plotly_dark')

fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.update_traces(textfont_size = 12, textangle = 0, textposition = "outside", cliponaxis = False)

fig.show()

# 这个是抄的kaggle，后面没弄这里有问题