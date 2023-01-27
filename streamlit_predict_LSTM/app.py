import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import json

st.title('Predict a curve with LSTM')

st.write('---')
tab1, tab2 = st.tabs(["predict", "code"])

with tab1:
    st.write('upload your table ')
    uploaded_file = st.file_uploader("Choose a file", type=["csv","txt"])

    if uploaded_file is not None:
        sep_select = st.selectbox('select your sep',
                                  (",", ";", "|", "\t"),
                                  index=0)
        df = pd.read_csv(uploaded_file,sep=sep_select)
        with st.expander("your data"):
            st.write(df[0:100])

        option = st.selectbox('select your column',(df.columns))

        col1, col2 = st.columns(2)
        test_size_select = 0.50
        with col1:
            train_size_select = st.slider('pourcent train',0.00,1.00,1.00-test_size_select,step=0.01)
        with col2:
            test_size_select = st.slider('pourcent test',0.00,1.00,1.00-train_size_select,step=0.01)

        look_back_select = st.number_input('look_back',min_value=1,max_value=len(df))
        epochs_select = st.number_input('epochs', min_value=1, max_value=200)

        star = st.button('Star')

        if star :

            #select the dataset
            df_stock = df[option]
            df_stock = df_stock.values[:].astype("float32")

            #reshape
            df_stock = df_stock.reshape(df_stock.shape[0], 1)

            #normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(df_stock)

            #select train and test
            train_size = int(len(dataset) * train_size_select)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

            # plot train and test
            fig, ax = plt.subplots()
            ax.plot(range(train_size, len(df_stock)), test)
            ax.plot(range(0, train_size), train)
            st.pyplot(fig)

            #fonction with look_back
            def create_dataset(dataset, look_back):
                dataX, dataY = [], []
                for i in range(len(dataset) - look_back):
                    a = dataset[i:(i + look_back), 0]
                    dataX.append(a)
                    dataY.append(dataset[i + look_back, 0])
                return np.array(dataX), np.array(dataY)

            #create X and Y for train and test
            trainX, trainY = create_dataset(train, look_back_select)
            testX, testY = create_dataset(test, look_back_select)

            #create model
            model = Sequential()
            model.add(LSTM(input_shape=(look_back_select, 1), units=100, return_sequences=True))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=1, activation='linear'))
            model.summary()
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

            my_bar = st.progress(0)

            #fit model
            with st.empty():
                for epoch in range(epochs_select):
                    history = model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
                    my_bar.progress(((epoch+1)*1)/epochs_select)
                    st.write(f"⏳ {epoch} epoch {epochs_select}")
                st.write("✔️ End!")

            #predict
            Predict = model.predict(testX)

            # plot predict
            fig, ax = plt.subplots()
            ax.plot(testY, label='true data')
            ax.plot(Predict, label="Predict data")
            st.pyplot(fig)

    with tab2:

        with open("LSTM_bourse.ipynb", mode="r", encoding="utf-8") as f:
            myfile = json.loads(f.read())

        for cell in myfile['cells']:
            code = ''
            for line in cell["source"]:
                code += line

            st.code(code, language='python')

       

















