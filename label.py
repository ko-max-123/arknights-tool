import os
import json
import cv2
import numpy as np
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape, Softmax, BatchNormalization, Dropout, Bidirectional, LSTM, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib

# フォント設定（日本語対応）
matplotlib.rcParams['font.family'] = 'MS Gothic'

# パラメータ設定
IMG_HEIGHT = 64
IMG_WIDTH = 256
MAX_LABEL_LEN = 32

# データ読み込みと前処理

def load_data(json_path, img_dir):
    images = []
    labels = []
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data['assets'].values():
            # パスの正規化
            img_path = os.path.normpath(os.path.join(img_dir, item['asset']['name']))
            # ファイル存在チェック
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0
            images.append(np.expand_dims(img, axis=-1))
            labels.append(item['regions'][0]['tags'][0])
    return np.array(images), np.array(labels)

# データ準備
base_dir = r'D:\github_dev\arknights-tool\dataset\train'
json_path = os.path.join(base_dir, 'tag', 'vott-json-export', 'tool_tag-export.json')
img_dir = os.path.join(base_dir, 'tag', 'vott-json-export')
images, labels = load_data(json_path, img_dir)

# データが空の場合は処理を中断
if len(images) == 0 or len(labels) == 0:
    raise ValueError("画像またはラベルが空です。データセットを確認してください。")

# 動的に文字セットを生成
CHAR_SET = ''.join(sorted(set(''.join(labels))))
num_classes = len(CHAR_SET) + 1  # 1つ追加して空白ラベル用

# ラベルのエンコーディング
label_to_index = {char: i + 1 for i, char in enumerate(CHAR_SET)}  # 1から始める
index_to_label = {i + 1: char for char, i in label_to_index.items()}
label_to_index['<BLANK>'] = 0  # 空白用ラベル

def encode_labels(labels):
    encoded = []
    for label in labels:
        encoded_label = [label_to_index[c] for c in label]
        # パディングを末尾に配置
        if len(encoded_label) < MAX_LABEL_LEN:
            encoded_label += [0] * (MAX_LABEL_LEN - len(encoded_label))
        encoded.append(encoded_label)
    return np.array(encoded)

encoded_labels = encode_labels(labels)

# 再度データ分割時にエラー防止
if len(images) < 2:  # データ数が極端に少ない場合はエラー
    raise ValueError("データ数が少なすぎます。追加のデータを準備してください。")

x_train, x_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# CTC損失関数
def ctc_loss_lambda_func(y_true, y_pred):
    input_length = np.expand_dims(np.ones(y_pred.shape[0]) * y_pred.shape[1], axis=1)
    label_length = np.expand_dims(np.sum(y_true > 0, axis=1), axis=1)  # パディングを除外した長さ
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# モデル設計
def build_model():
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Reshape(target_shape=(32, -1))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # モデル構築
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=ctc_loss_lambda_func)
    return model

model = build_model()

# 学習
epochs = 100
batch_size = 32
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test)
)

# モデル保存
model.save('text_recognition_model.keras')
