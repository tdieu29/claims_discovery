import sqlite3 
import streamlit as st

def show_all():

    # Display
    st.set_page_config(
    page_title="Demo...???",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )

    options = st.sidebar.radio("Type of resulting articles to display",
                                ["Supporting", "Contradicting", "Not enough information"])
    
    st.subheader("Enter a query or a claim")
    query = st.text_input(label="Write sth here....................")
    st.button('Search')

    db = sqlite3.connect('cord19_data/database/articles.sqlite')
    cur = db.cursor()  
    count = 1
    idx_list = [1, 2]
    
    for idx in idx_list:
        article_id = cur.execute("SELECT Article_Id FROM sections WHERE Section_Id = (?)", (idx,)).fetchone()[0]
        abstract = cur.execute("SELECT Abstract FROM articles WHERE Article_Id = (?)", (article_id,)).fetchone()[0] 
        title =  cur.execute("SELECT Title FROM articles WHERE Article_Id = (?)", (article_id,)).fetchone()[0]  
        published_date = cur.execute("SELECT Published_Date FROM articles WHERE Article_Id = (?)", (article_id,)).fetchone()[0] 
        authors = cur.execute("SELECT Authors FROM articles WHERE Article_Id = (?)", (article_id,)).fetchone()[0] 
        journal = cur.execute("SELECT Journal FROM articles WHERE Article_Id = (?)", (article_id,)).fetchone()[0] 
        url = cur.execute("SELECT Url FROM articles WHERE Article_Id = (?)", (article_id,)).fetchone()[0] 

        st.subheader(f'{count}. [{title}]({url})')
        st.write(f"{authors}. <em>{journal}.</em> ({published_date})", unsafe_allow_html=True)
        
        sentence_indexes = [1, 2, 3]
        for i in sentence_indexes:
            sentence = cur.execute("SELECT Sentence FROM sentences WHERE Article_Id = (?) AND Sentence_Index = (?)", (article_id, i)).fetchone()[0]
            st.write(f'-  {sentence}')

        if st.checkbox('Show full abstract', key=count):
            st.write(f'{abstract}')
        
        count += 1

    db.close()


show_all()