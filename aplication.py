#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import re

st.set_page_config(layout='wide')
url = 'https://image.tmdb.org/t/p/original'
dataset_ML = pd.read_csv(r'C:\Users\Fernando\Documents\Project2\df_dummies.csv', sep=',')


# In[9]:
st.markdown( """<h1 style="text-align: center;">TITLE SITE</h1>""", unsafe_allow_html=True,)
st.write("")

with st.sidebar:
    with st.form("filters"):
        film_name = st.text_input("Enter the name of your film").lower()
        film_name = re.sub(r"[:',-]", " ", film_name)
        film_name = re.sub(r"\s+", " ", film_name)
        years = st.slider("Release date", dataset_ML["startYear"].min(), dataset_ML["startYear"].max(), (dataset_ML["startYear"].min(),     dataset_ML["startYear"].max()))

        submitted = st.form_submit_button("Submit")
        if submitted:

            X = dataset_ML.select_dtypes(include='number')
            y = dataset_ML['primaryTitle']
            
            #Standardisation des variables explicatives 
            X_scaled = StandardScaler().fit_transform(X)       

            #Calculo de indices en donde se encuentran las columnas de actores, directores y generos
            genres1 = ['Action', 'Adventure', 'Animation', 'Sci-Fi']
            genres2 = ['Comedy', 'Crime', 'Drama', 'Family', 'Film-Noir', 'Horror', 'Romance']
            posiciones_genres1 = [X.columns.get_loc(col) for col in genres1]
            posiciones_genres2 = [X.columns.get_loc(col) for col in genres2]
            posiciones_actor = [X.columns.get_loc(col) for col in X.columns if col.startswith('actor')]
            posiciones_actress = [X.columns.get_loc(col) for col in X.columns if col.startswith('actress')]
            posiciones_directors = [X.columns.get_loc(col) for col in X.columns if col.startswith('director')]
            posiciones_actors = posiciones_actor + posiciones_actress

            X_scaled[:, posiciones_genres1] *= 20
            X_scaled[:, posiciones_genres2] *= 2
            X_scaled[:, posiciones_actors] *= 20
            X_scaled[:, posiciones_directors] *= 30

            #formation du modèle
            model = KNeighborsClassifier(n_neighbors=25, weights='distance').fit(X_scaled, y)
            
            #Recherche de l'indice du film recherché et des films recommandès
            dataset_ML['primaryTitle_low'] = dataset_ML['primaryTitle'].apply(lambda x: re.sub(r"\s+", " ", re.sub(r"[:',-]", " ", x.lower())).strip())
            #Voy a modifica la siguiente linea para que al realizar la busqueda no importe si se escribe mal
            #idx = dataset_ML.loc[dataset_ML['primaryTitle_low'] == film_name].index
            idx = dataset_ML.loc[dataset_ML['primaryTitle_low'].str.contains(film_name)].index
            if not idx.empty:
                index_movie = idx[0]
                recom = model.kneighbors([X_scaled[index_movie]])[1][0]
                dataset_recom = dataset_ML.iloc[recom]
                dataset_recom = dataset_recom.loc[dataset_recom.startYear.between(years[0], years[1])]
            else:
                st.write("Vérifie le nom du film ou essaie avec un autre")
                dataset_recom = dataset_ML.loc[dataset_ML.startYear.between(years[0], years[1])]
        else:
            dataset_recom = dataset_ML.loc[dataset_ML.startYear.between(years[0], years[1])]

if not dataset_recom.empty:
    row1, row2, row3, row4, row5 = st.columns(3), st.columns(3), st.columns(3), st.columns(3), st.columns(3)
                
    for i, col in zip(range(len(dataset_recom)), row1 + row2 + row3 + row4 + row5):
        with col.container():
            col1, col2 = st.columns(2)
                        
            with col1:
                st.image(url + dataset_recom.iloc[i]['poster_path'], use_column_width="auto")
                        
            with col2:
                st.header(dataset_recom.iloc[i]['primaryTitle'])
                st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']))
                st.write("Realease Date: " + str(dataset_recom.iloc[i]['startYear']))
                st.write("Runtime: " + str(dataset_recom.iloc[i]['runtimeMinutes']) + "min")
else:
    dataset_recom = dataset_ML





