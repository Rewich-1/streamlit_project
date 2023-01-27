import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import json

st.title('Predict clothes with fashion_mnist')

st.write('---')
tab1, tab2 ,tab3= st.tabs(["train","predict", "code"])

with tab1:
    st.write('upload your table ')
    with st.spinner("Please wait..."):
        data = keras.datasets.fashion_mnist
        cifar10_data = data.load_data()

    (train_images, train_labels), (test_images, test_labels) = cifar10_data
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    with st.expander("Look the dataset"):
        train_index = st.number_input('Insert a number', min_value=0, max_value=len(train_images))

        fig, ax = plt.subplots()
        ax.imshow(train_images[train_index])
        st.pyplot(fig)
        st.header(class_names[train_labels[train_index]])

    epochs_select = st.number_input('epochs', min_value=1, max_value=200)
    save_model = st.checkbox('save model')


    star = st.button('Star')
    if star:

        # normalize the dataset
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        st.write(train_images.shape)
        st.write(test_images.shape)

        # create model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(254, activation='relu'),
            keras.layers.Dense(10, activation='softmax')])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        my_bar = st.progress(0)

        # fit model
        dic = []
        with st.empty():
            for epoch in range(epochs_select):
                st.write(f"⏳ {epoch} epoch {epochs_select}")

                model.fit(train_images, train_labels, epochs=1)
                test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
                dic.append(test_acc)

                my_bar.progress(((epoch + 1) * 1) / epochs_select)

            st.write("✔️ End!")

        fig, ax = plt.subplots()
        ax.plot(range(0,len(dic)), dic)
        ax.set_title('eval')
        st.pyplot(fig)

        st.write(train_images.shape)
        st.write(test_images.shape)
        predictions = model.predict (test_images)


        if save_model:
            model.save('my_model.h5')
            del model

with tab2:



    index_test = st.number_input('Insert index', min_value=0, max_value=len(test_images))

    model = load_model('my_model.h5')


    fig, ax = plt.subplots()
    ax.imshow(test_images[index_test])
    st.pyplot(fig)
    st.header("label : " + class_names[train_labels[index_test]])

    predictions = model.predict(train_images)
    st.header("label predict : " + class_names[int(np.argmax(predictions[index_test]))])


with tab3:

    with open("tp1.ipynb", mode="r", encoding="utf-8") as f:
        myfile = json.loads(f.read())

    for cell in myfile['cells']:
        code = ''
        for line in cell["source"]:
            code += line

        st.code(code, language='python')

       

















