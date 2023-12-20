import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Загружаем предобученную модель InceptionV3
model = InceptionV3(weights='imagenet')

def predict_church(image_path):
    # Загружаем изображение
    img = image.load_img(image_path, target_size=(299, 299))

    # Преобразуем изображение в массив numpy и дополняем его до 4D
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Предварительная обработка изображения для модели InceptionV3
    img_array = preprocess_input(img_array)

    # Получаем предсказания
    predictions = model.predict(img_array)

    # Декодируем и выводим топ-3 предсказаний
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def predict_and_display(image_path):
    # Загружаем изображение
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Предсказываем содержание изображения
    predictions = predict_church(image_path)
    st.write("## Результаты распознавания:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

# Веб-приложение с использованием Streamlit
st.title("Распознавание храмов и церквей")

uploaded_file = st.file_uploader("Загрузите изображение храма или церкви", type="jpg")

if uploaded_file is not None:
    # Сохраняем изображение
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Выводим изображение и результаты распознавания
    predict_and_display(image_path)
