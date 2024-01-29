#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(layout='wide')
url = 'https://image.tmdb.org/t/p/original'
dataset_ML = pd.read_csv(r'C:\Users\Fernando\Documents\Project2\df_dummies.csv', sep=',')


# In[9]:
st.markdown( """<h1 style="text-align: center;">TITLE SITE</h1>""", unsafe_allow_html=True,)
st.write("")

with st.sidebar:
    with st.form("filters"):
        film_name = st.text_input("Enter the name of your film")
        years = st.slider("Release date", dataset_ML["startYear"].min(), dataset_ML["startYear"].max(), (dataset_ML["startYear"].min(),     dataset_ML["startYear"].max()))

        submitted = st.form_submit_button("Submit")
        if submitted:

            X = dataset_ML.select_dtypes(include='number')
            y = dataset_ML['primaryTitle']
            
            #Standardisation des variables explicatives et formation du modèle
            X_scaled = StandardScaler().fit_transform(X)
            model = KNeighborsClassifier(n_neighbors=10, weights='distance').fit(X_scaled, y)
            
            #Recherche de l'indice du film recherché et des films recommandès
            
            index_movie = dataset_ML.loc[dataset_ML['primaryTitle'] == film_name].index[0]
            recom = model.kneighbors([X_scaled[index_movie]])[1][0]
            dataset_recom = dataset_ML.iloc[recom]
    
            dataset_recom = dataset_recom.loc[dataset_recom.startYear.between(years[0], years[1])]

row1, row2, row3, row4 = st.columns(3), st.columns(3), st.columns(3), st.columns(3)
            
for i, col in zip(range(len(dataset_recom)), row1 + row2 + row3 + row4):
    with col.container():
        col1, col2 = st.columns(2)
                    
        with col1:
            st.image(url + dataset_recom.iloc[i]['poster_path'], use_column_width="auto")
                    
        with col2:
            st.header(dataset_recom.iloc[i]['primaryTitle'])
            st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']))
            st.write("Realease Date: " + str(dataset_recom.iloc[i]['startYear']))
            st.write("Runtime: " + str(dataset_recom.iloc[i]['runtimeMinutes']) + "min")

# In[ ]:





