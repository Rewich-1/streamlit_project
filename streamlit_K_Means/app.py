import streamlit as st
import json

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np


st.title('how works K_Means')

st.write('---')
tab1, tab2 = st.tabs(["predict", "code"])

with tab1:

    st.write('Configure your dataset')

    col1, col2, col3 = st.columns(3)
    with col1:
        nb_samples = st.slider('nombre samples',0,100,100)
    with col2:
        nb_center = st.slider('nombre centers',0,10,3)
    with col3:
        std = st.slider('cluster_std', 0.0, 1.0, 0.5,step=0.1)

    nb_clusters = st.slider('n_clusters ', 1, 10, 2)



    col1, col2 = st.columns(2)
    with col1:
        #generate dataset
        X , y = make_blobs(n_samples = nb_samples , centers =  nb_center , cluster_std= std , random_state = 0 )

        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1])
        st.pyplot(fig)


    with col2:

        # fit
        model = KMeans(n_clusters=nb_clusters, random_state=0).fit(X)

        # predictions
        predictions = model.labels_

        #plot scluster
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=predictions)
        st.pyplot(fig)



    with tab2:

        with open("K_Means.ipynb", mode="r", encoding="utf-8") as f:
            myfile = json.loads(f.read())

        for cell in myfile['cells']:
            code = ''
            for line in cell["source"]:
                code += line

            st.code(code, language='python')

       

















