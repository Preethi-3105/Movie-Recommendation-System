import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
#from streamlit.cli import main
import streamlit as st

data=pd.read_csv('/home/tilak/MCAD/IMDb movies.csv',low_memory = False)
#movie = "The Avengers"
movie = st.text_input("Enter your movie")
data = data[data['votes']>data['votes'].mean()]
#print(data.info())
data['desp']=data['director'].astype(str) + "\n" + data['writer'].astype(str) + "\n" + data['production_company'].astype(str) + "\n" + data['actors'].astype(str) + "\n" + data['description'].astype(str)
tfidf_vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1, 2), max_df=0.8, token_pattern=r'\b[^\d\W][^\d\W]+\b',min_df=10)
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in data["desp"]])
cosine_similarity_df=cosine_similarity(tfidf_matrix)
cosine_simialarity=pd.DataFrame(cosine_similarity_df,index=data.title,columns=data.title)
#print(cosine_simialarity.head())
Choice=cosine_simialarity.loc[:,movie]
Ordered_similarities=Choice.sort_values(ascending=False)
Recommendations=pd.DataFrame(Ordered_similarities)
st.table(Recommendations.iloc[1:11].index)
#print("The selected movie is ",movie)