import streamlit as st
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


st.title('how works K_Means')

st.write('---')
tab1, tab2 = st.tabs(["predict", "code"])

with tab1:

    st.write('Configure your dataset')

    col1, col2, col3 = st.columns(3)
    with col1:
        nb_samples = st.slider('nombre samples',0,100,100)
    with col2:
        select_X = st.slider('change x ', 1, 100, 2)
    with col3:
        select_Y = st.slider('change y', 0.0, 1.0, 0.5,step=0.1)

    axe = st.slider('axe', 0, 100, 100)




    col1, col2 = st.columns(2)
    with col1:

        #generate dataset
        rnstate = np.random.RandomState(1)
        x = select_X * rnstate.rand(nb_samples)
        y = select_Y * x + rnstate.randn(nb_samples)

        fig, ax = plt.subplots()
        plt.scatter(x, y)
        st.pyplot(fig)


    with col2:

        # fit
        model = LinearRegression(fit_intercept=True)
        model.fit(x[:, np.newaxis], y)


        # predictions
        xfit = np.linspace(0, select_X, 1000)
        yfit = model.predict(xfit[:, np.newaxis])

        #plot graph
        fig, ax = plt.subplots()
        plt.scatter(x, y)
        plt.plot(xfit, yfit)
        st.pyplot(fig)

        # print coef
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('coef: ', model.coef_[0])
    with col2:
        st.write('intercept: ', model.intercept_)
    with col3:
        st.write('R2: ', model.score(x[:, np.newaxis], y))




    with tab2:

        with open("LinearRegression.ipynb", mode="r", encoding="utf-8") as f:
            myfile = json.loads(f.read())

        for cell in myfile['cells']:
            code = ''
            for line in cell["source"]:
                code += line

            st.code(code, language='python')

       

















