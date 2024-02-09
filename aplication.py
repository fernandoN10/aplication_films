# -*- coding: utf-8 -*-
"""projet2streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rV6PV5sB5n4X3sMKzC1Lq1IlPqLE1Zdc
"""

# -*- coding: utf-8 -*-
#"""projet2streamlit.ipynb

#Automatically generated by Colaboratory.

#Original file is located at
#    https://colab.research.google.com/drive/1TGrNLxVXD1K1fw_SApgjAjlnwYU5QPgf
#"""

#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler
from fuzzywuzzy import fuzz
import re


url = 'https://image.tmdb.org/t/p/original'
dataset_ML = pd.read_csv(r'df_dummies.csv', sep=',')
dataset_ML['cle'] = dataset_ML['liste_actors'] + dataset_ML['liste_actress'] + dataset_ML['liste_director'] + dataset_ML['genres']

# In[9]:
#st.markdown( """<h1 style="text-align: center;">TITLE SITE</h1>""", unsafe_allow_html=True,)
st.set_page_config(page_title="WILDFLIX", page_icon="", layout="wide")

col1, col2, col3 = st.columns(3)
with col1:
    st.write('')
with col2:
    st.image("logo.png", use_column_width="auto")
with col3:
    st.write('')
st.write('')

with st.sidebar:
    with st.form("filters"):
        liste_films = tuple([' '] + dataset_ML['primaryTitle'].to_list())
        film_name = st.selectbox('What movie would you like to see?',liste_films)

        list_genres = st.multiselect('Choose one or more genres', dataset_ML['genres'].str.split(',').explode().dropna().unique().tolist(), [])

        # list_companies = st.multiselect('Choose one or more companies', dataset_ML['companies_names'].str.split(', ').explode().dropna().unique().tolist(), [])
        list_companies = st.multiselect('Choose one or more companies', list(set(dataset_ML.columns[578:614])), [])

        combined_list = list(set(dataset_ML.columns[67:577]))
        list_names = st.multiselect('Choose one or more names', combined_list, [])

        #years = st.slider("Release date", dataset_ML["startYear"].min(), dataset_ML["startYear"].max(), (dataset_ML["startYear"].min(),     dataset_ML["startYear"].max()))

        submitted = st.form_submit_button("Submit")
if submitted:

    if film_name != ' ':

        dataset_ML['title_similary'] = dataset_ML['primaryTitle'].apply(lambda x: fuzz.ratio(film_name, x))

        NL = dataset_ML.loc[dataset_ML['primaryTitle'] == film_name, 'cle'].to_string(index=False)
        dataset_ML['NL'] = dataset_ML['cle'].apply(lambda x: fuzz.ratio(NL, x))


        X = dataset_ML.select_dtypes(include='number')
        #y = dataset_ML['primaryTitle']

        #Standardisation des variables explicatives
        X_scaled = StandardScaler().fit_transform(X)
        #X_scaled = RobustScaler().fit_transform(X)

        liste_cols = X.columns.to_list()

        film_selecctione = dataset_ML[dataset_ML['primaryTitle'] == film_name]
        liste_genres_bonus = ['Animation', 'Family', 'History', 'Horror', 'Western']
        liste_genres_film = film_selecctione['genres'].iloc[0].split(',')
        posiciones_genres = [X.columns.get_loc(col) for col in liste_genres_film]
        posiciones_genres_bonus = [X.columns.get_loc(col) for col in liste_genres_bonus]

        if film_selecctione['liste_actors'].iloc[0] != ' ':
            actors = film_selecctione['liste_actors'].iloc[0]
            actors = actors.split(',')
        else:
            actors = []

        if film_selecctione['liste_actress'].iloc[0] != ' ':
            actresses = film_selecctione['liste_actress'].iloc[0]
            actresses = actresses.split(',')
        else:
            actresses = []

        if film_selecctione['liste_director'].iloc[0]:
            directors = film_selecctione['liste_director'].iloc[0]
            directors = directors.split(',')
        else:
            directors = []

        posiciones_actors = []
        for col in liste_cols:
            for actor in actors:
                if actor in col:
                    print("Actor", actor)
                    posiciones_actors.append(liste_cols.index(col))
                else:
                    posiciones_actors = posiciones_actors
            for actress in actresses:
                if actress in col:
                    print("Actress", actress)
                    posiciones_actors.append(liste_cols.index(col))
                else:
                    posiciones_actors = posiciones_actors

        posiciones_director = []
        for col in liste_cols:
            for director in directors:
                if director in col:
                    posiciones_director.append(liste_cols.index(col))

        #X_scaled[:, posiciones_genres_bonus] *= 9

        #if list_genres != []:
            #genres_filtre = list_genres.split(',')
            #position_genre_filtre = [X.columns.get_loc(col) for col in genres_filtre]
        #    position_genre_filtre = [X.columns.get_loc(col) for col in list_genres]
        #else:
        #    position_genre_filtre = []

        #if list_companies != []:
            #companies_filtre = list_companies.split(',')
        #    position_companies_filtre = [X.columns.get_loc(col) for col in list_companies]
        #else:
        #    position_companies_filtre = []

        #if list_names != []:
            #names_filtre = list_names.split(',')
        #    position_names_filtre = [X.columns.get_loc(col) for col in list_names]
        #else:
        #    position_names_filtre = []

        idx = dataset_ML.loc[dataset_ML['primaryTitle'] == film_name].index
        if not idx.empty:
            index_movie = idx[0]
            X_scaled_recom = X_scaled[index_movie]

        positions_film = posiciones_genres + posiciones_actors + posiciones_director + [X.columns.get_loc('NL')] + [X.columns.get_loc('title_similary')] #+ position_genre_filtre + position_companies_filtre + position_names_filtre
        positions_others = [i for i in range(len(liste_cols)) if i not in positions_film]

        #X_scaled[:, position_genre_filtre] *= 100
        #X_scaled[:, position_companies_filtre] *= 1
        #X_scaled[:, position_names_filtre] *= 1

        X_scaled[:, positions_others] *= 1
        X_scaled[:, posiciones_genres_bonus] *= 5
        X_scaled[:, posiciones_genres] *= 3
        X_scaled[:, posiciones_actors] *= 3
        X_scaled[:, posiciones_director] *= 2
        #X_scaled[:, posiciones_startYear] *= 1.5
        X_scaled[:, X.columns.get_loc('NL')] *= 2
        X_scaled[:, X.columns.get_loc('title_similary')] *= 3
        #X_scaled[:, X.columns.get_loc('overview_num')] *= 2

        # Ajout poids genres dans modèle.
        if not list_genres == [] :
            for genre in list_genres:
                id_col = X.columns.get_loc(genre)
                X_scaled_recom[id_col] = abs(X_scaled_recom[id_col])
                for i in range(len(X_scaled)) :
                    X_scaled[i][id_col] = X_scaled[i][id_col] * 100

        # Ajout poids compagnies dans modèle.
        if not list_companies == [] :
            for company in list_companies:
                id_col = X.columns.get_loc(company)
                X_scaled_recom[id_col] = abs(X_scaled_recom[id_col])
                for i in range(len(X_scaled)) :
                    X_scaled[i][id_col] = X_scaled[i][id_col] * 100

        # Ajout poids names.
        if not list_names == [] :
            for name in list_names:
                id_col = X.columns.get_loc(name)
                X_scaled_recom[id_col] = abs(X_scaled_recom[id_col])
                for i in range(len(X_scaled)) :
                    X_scaled[i][id_col] = X_scaled[i][id_col] * 1000


        cosine_similarity_matrix = cosine_similarity(X_scaled)

        model = NearestNeighbors(n_neighbors=50, metric='cosine').fit(X_scaled)
        recom = model.kneighbors([X_scaled[index_movie]])[1][0]
        dataset_recom = dataset_ML.iloc[recom]

        #else:

            #dataset_recom = dataset_ML.loc[dataset_ML.startYear.between(years[0], years[1])]

    else:
        #st.write("Vérifie le nom du film ou essaie avec un autre")
        #dataset_recom = dataset_ML.sort_values(by='notePondere', ascending=False)
                # Ajout poids genres dans modèle.
        X = dataset_ML.select_dtypes(include='number')
        liste_cols = X.columns.to_list()        

        #Standardisation des variables explicatives
        X_scaled = StandardScaler().fit_transform(X)

        posiciones_genres_list = []
        for col in liste_cols:
            for genre in list_genres:
                if genre in col:
                    posiciones_genres_list.append(liste_cols.index(col))
        posiciones_companies_list = []
        for col in liste_cols:
            for company in list_companies:
                if company in col:
                    posiciones_companies_list.append(liste_cols.index(col))
        posiciones_names_list = []
        for col in liste_cols:
            for name in list_names:
                if name in col:
                    posiciones_names_list.append(liste_cols.index(col))
       
        positions_vector_filtres = posiciones_genres_list + posiciones_companies_list + posiciones_names_list
        positions_others_vector = [i for i in range(len(liste_cols)) if i not in positions_vector_filtres]
        positios_vector = positions_vector_filtres + positions_others_vector
        vector_long = max(positios_vector) + 1

        vector = [0] * vector_long

        for pos in positions_vector_filtres:
            vector[pos] = 10
        for pos in positions_others_vector:
            vector[pos] = 0

        X_scaled[:, positions_vector_filtres] *= 10
        X_scaled[:, positions_others_vector] *= -1

        vector = np.array(vector)
        similarity = cosine_similarity(vector.reshape(1, -1), X_scaled)

        model = NearestNeighbors(n_neighbors=50).fit(X_scaled)
        
        #distance = np.linalg.norm(X_scaled - vector, axis=1)
        index_closer = np.argmax(similarity)
        #cosine_similarity_matrix = cosine_similarity(X_scaled)
                
        recom = model.kneighbors([X_scaled[index_closer]])[1][0]
        dataset_recom = dataset_ML.iloc[recom]
        dataset_recom = dataset_recom.sort_values(by='notePondere, ascending=False)

    with st.container():
        st.subheader('', divider='gray')

        col1, col2, col3, col4 = st.columns(4)

        with col2:
            st.image(url + dataset_recom.iloc[0]['poster_path'], width=300, use_column_width="auto")
            #

        with col3:
            st.header(dataset_recom.iloc[0]['primaryTitle'])
            if dataset_recom.iloc[0]['liste_actress'] == ' ':
                st.write("Actors: " + str(dataset_recom.iloc[0]['liste_actors']))
            else:
                st.write("Actors: " + str(dataset_recom.iloc[0]['liste_actors']) + ", "  + str(dataset_recom.iloc[0]['liste_actress']))
            st.write("Director: " + str(dataset_recom.iloc[0]['liste_director']))
            st.write("Genres: " + str(dataset_recom.iloc[0]['genres']))
            st.write("Realease Date: " + str(dataset_recom.iloc[0]['startYear']))
            st.write("Runtime: " + str(dataset_recom.iloc[0]['runtimeMinutes']) + "min")
            st.write("Note Imdb: " + str(dataset_recom.iloc[0]['averageRating']))

            st.write('')
            st.write('')

        #    with col3:
        #        st.write('')

        with st.container():
            st.subheader('Recomended for you:', divider='gray')
            row1, row2, row3, row4, row5 = st.columns(3), st.columns(3), st.columns(3), st.columns(3), st.columns(3)

            for i, col in zip(range(len(dataset_recom)), row1 + row2 + row3 + row4 + row5):

                with col.container(border=True):
                    st.image(url + dataset_recom.iloc[i+1]['poster_path'], use_column_width="auto")
                    st.header(dataset_recom.iloc[i+1]['primaryTitle'])
                    if dataset_recom.iloc[i+1]['liste_actress'] == ' ':
                        st.write("Actors: " + str(dataset_recom.iloc[i+1]['liste_actors']))
                    else:
                        st.write("Actors: " + str(dataset_recom.iloc[i+1]['liste_actors']) + ", "  + str(dataset_recom.iloc[i+1]['liste_actress']))
                    st.write("Director: " + str(dataset_recom.iloc[i+1]['liste_director']))
                    st.write("Genres: " + str(dataset_recom.iloc[i+1]['genres']))
                    st.write("Realease Date: " + str(dataset_recom.iloc[i+1]['startYear']))
                    st.write("Runtime: " + str(dataset_recom.iloc[i+1]['runtimeMinutes']) + "min")
                    st.write("Note Imdb: " + str(dataset_recom.iloc[i+1]['averageRating']))

if not submitted:

    dataset_recom = dataset_ML.sort_values(by='notePondere', ascending=False)

    with st.container():
        st.subheader('Recomended for you:', divider='gray')
        row1, row2, row3, row4, row5 = st.columns(4), st.columns(4), st.columns(4), st.columns(4), st.columns(4)

        for i, col in zip(range(len(dataset_recom)), row1 + row2 + row3 + row4 + row5):

            with col.container(border=True):
                st.image(url + dataset_recom.iloc[i]['poster_path'], use_column_width="auto")
                st.header(dataset_recom.iloc[i]['primaryTitle'])
                if dataset_recom.iloc[i]['liste_actress'] == ' ':
                    st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']))
                else:
                    st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']) + ", "  + str(dataset_recom.iloc[i]['liste_actress']))
                st.write("Director: " + str(dataset_recom.iloc[i]['liste_director']))
                st.write("Genres: " + str(dataset_recom.iloc[i]['genres']))
                st.write("Realease Date: " + str(dataset_recom.iloc[i]['startYear']))
                st.write("Runtime: " + str(dataset_recom.iloc[i]['runtimeMinutes']) + "min")
                st.write("Note Imdb: " + str(dataset_recom.iloc[i]['averageRating']))


#else:

    # with st.container():
    #     st.subheader('Top Films:', divider='gray')
    #     row1, row2, row3, row4, row5, row6, row7, row8 = st.columns(4), st.columns(4), st.columns(4), st.columns(4), st.columns(4), st.columns(4), st.columns(4), st.columns(4)

    #     for i, col in zip(range(len(dataset_recom)), row1 + row2 + row3 + row4 + row5+ row6 + row7 + row8):

    #         with col.container(border=True):
    #             st.image(url + dataset_recom.iloc[i]['poster_path'], use_column_width="auto")
    #             st.header(dataset_recom.iloc[i]['primaryTitle'])
    #             if dataset_recom.iloc[i]['liste_actress'] == ' ':
    #                 st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']))
    #             else:
    #                 st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']) + ", "  + str(dataset_recom.iloc[i]['liste_actress']))
    #             st.write("Director: " + str(dataset_recom.iloc[i]['liste_director']))
    #             st.write("Genres: " + str(dataset_recom.iloc[i]['genres']))
    #             st.write("Realease Date: " + str(dataset_recom.iloc[i]['startYear']))
    #             st.write("Runtime: " + str(dataset_recom.iloc[i]['runtimeMinutes']) + "min")
    #             st.write("Note Imdb: " + str(dataset_recom.iloc[i]['averageRating']))


#            with col2:
#                st.header(dataset_recom.iloc[i]['primaryTitle'])
#                st.write("Actors: " + str(dataset_recom.iloc[i]['liste_actors']) + ", " + str(dataset_recom.iloc[i]['liste_actress']))
#                st.write("Director: " + str(dataset_recom.iloc[i]['liste_director']))
#                st.write("Genres: " + str(dataset_recom.iloc[i]['genres']))
#                st.write("Realease Date: " + str(dataset_recom.iloc[i]['startYear']))
#                st.write("Runtime: " + str(dataset_recom.iloc[i]['runtimeMinutes']) + "min")
#                st.write("Note Imdb: " + str(dataset_recom.iloc[i]['averageRating']))
