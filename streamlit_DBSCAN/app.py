import streamlit as st
import json
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN


st.title('how works DBSCAN')

st.write('---')
tab1, tab2 = st.tabs(["predict", "code"])

with tab1:

    st.write('Configure your dataset')

    col1, col2, col3 = st.columns(3)
    with col1:
        nb_samples = st.slider('nombre samples',0,100,10)
    with col2:
        nb_center = st.slider('nombre centers',0,100,10)
    with col3:
        std = st.slider('cluster_std', 0.0, 1.0, 0.5,step=0.1)


    col1, col2 = st.columns(2)
    with col1:
        eps_select = st.slider('eps select', 0.0, 1.0, 0.8, step=0.1)
    with col2:
        nb_min_samples = st.slider('min_samples ', 0, 10, 2)


    col1, col2 = st.columns(2)
    with col1:
        #generate dataset
        X, y = make_blobs(n_samples=nb_samples, centers=nb_center, cluster_std=std, random_state=0)
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1])
        st.pyplot(fig)


    with col2:
        #fit
        model = DBSCAN(eps=eps_select, min_samples=nb_min_samples)
        model.fit(X)

        #create label
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        labels = model.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        #plot scluster
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=model.fit_predict(X))
        st.pyplot(fig)
        st.write("Estimated number of clusters: %d" % n_clusters_)


    with tab2:

        with open("DBSCAN.ipynb", mode="r", encoding="utf-8") as f:
            myfile = json.loads(f.read())

        for cell in myfile['cells']:
            code = ''
            for line in cell["source"]:
                code += line

            st.code(code, language='python')

       

















