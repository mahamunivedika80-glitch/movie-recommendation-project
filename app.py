import pandas as pd
import streamlit as st
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Title
# -------------------------------
st.title("Movie Recommender System")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("cleaned_data.csv")

# -------------------------------
# Generate similarity matrix if it doesn't exist
# -------------------------------
if not os.path.exists("similarity.pkl"):
    if st.button("Generate Similarity"):
        cv = CountVectorizer(max_features=10000, stop_words="english")

        dtm = cv.fit_transform(df["tags"])

        similarities = cosine_similarity(dtm)

        with open("similarity.pkl", "wb") as file:
            pickle.dump(similarities, file)

        st.success("Similarity matrix generated successfully!")

# -------------------------------
# Load similarity matrix
# -------------------------------
if os.path.exists("similarity.pkl"):
    with open("similarity.pkl", "rb") as file:
        similarities = pickle.load(file)
else:
    similarities = None

# -------------------------------
# Movie List
# -------------------------------
movies = df["title"].tolist()

name = st.selectbox("Select a movie", movies)

# -------------------------------
# Helper Functions
# -------------------------------
def get_name_by_index(i):
    if 0 <= i < len(df):
        return df.loc[i, "title"]
    return ""


def get_index_from_name(name):
    clean_name = name.lower().replace(" ", "").replace("-", "")

    for i in df.index:
        movie = (
            df.loc[i, "title"]
            .lower()
            .replace(" ", "")
            .replace("-", "")
        )

        if movie == clean_name:
            return i

    return -1


# -------------------------------
# Recommendation Button
# -------------------------------
if st.button("Recommend"):

    if similarities is None:
        st.error("Please generate the similarity matrix first.")
    else:
        index = get_index_from_name(name)

        if index == -1:
            st.error("Movie not found.")
        else:
            similarity_scores = list(enumerate(similarities[index]))
            similarity_scores = sorted(
                similarity_scores,
                key=lambda x: x[1],
                reverse=True
            )

            st.subheader("Top 5 Recommended Movies")

            count = 0
            for movie in similarity_scores:
                if movie[0] != index:
                    st.write(f"⭐ {get_name_by_index(movie[0])}")
                    count += 1

                if count == 5:
                    break