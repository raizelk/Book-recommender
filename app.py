import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models and data
collab_model = pickle.load(open('model.pkl', 'rb'))
book_names = pickle.load(open('book_name.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))
final_rating = pickle.load(open('rating_books.pkl', 'rb'))
unibook = pickle.load(open('unibook.pkl', 'rb'))
books = pd.DataFrame(unibook)
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.header('Combined Book Recommender System')

def fetch_poster(suggestion):
    book_names = []
    ids_index = []
    poster_urls = []

    for book_id in suggestion:
        book_names.append(book_pivot.index[book_id])

    for name in book_names:
        ids = np.where(final_rating['title'] == name)[0]
        if len(ids) > 0:
            ids_index.append(ids[0])

    for idx in ids_index:
        url = final_rating.iloc[idx]['img']
        poster_urls.append(url)

    return poster_urls

def recommend_book_collaborative(book_name):
    """Collaborative filtering recommendations"""
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0]

    if len(book_id) == 0:
        st.write(f"Book '{book_name}' not found in book_pivot index.")
        return [], []

    book_id = book_id[0]
    distances, suggestions = collab_model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_urls = fetch_poster(suggestions[0])

    for i in range(1, len(suggestions[0])):
        suggested_book_id = suggestions[0][i]
        books_list.append(book_pivot.index[suggested_book_id])

    return books_list, poster_urls

def recommend_book_content_based(book_name):
    """Content-based filtering recommendations"""
    if book_name not in books['title'].values:
        st.write(f"Book '{book_name}' not found in books DataFrame.")
        return [], []

    book_index = books[books['title'] == book_name].index[0]
    distances = similarity[book_index]
    book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_books = []
    recommended_books_posters = []
    for i in book_list:
        recommended_books.append(books.iloc[i[0]].title)
        recommended_books_posters.append(books.iloc[i[0]]['img'])

    return recommended_books, recommended_books_posters

selected_book_name = st.selectbox(
    "Select a book:",
    books['title'].values
)

if st.button('Show Recommendation'):
    # Get recommendations from both methods
    recommended_books_collab, poster_urls_collab = recommend_book_collaborative(selected_book_name)
    recommended_books_content, poster_urls_content = recommend_book_content_based(selected_book_name)

    # Display recommendations from collaborative filtering
    st.subheader('Collaborative Filtering Recommendations')
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, col in enumerate([col1, col2, col3, col4, col5]):
        if i < len(recommended_books_collab):
            col.text(recommended_books_collab[i])
            col.image(poster_urls_collab[i])

    # Display recommendations from content-based filtering
    st.subheader('Content-Based Filtering Recommendations')
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, col in enumerate([col1, col2, col3, col4, col5]):
        if i < len(recommended_books_content):
            col.text(recommended_books_content[i])
            col.image(poster_urls_content[i])

    if not recommended_books_collab and not recommended_books_content:
        st.write("No recommendations available for this book.")
