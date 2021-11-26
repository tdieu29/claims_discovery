import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import streamlit as st

sys.path.insert(1, Path(__file__).parent.parent.absolute().__str__())

from colbert.retrieve import retrieve_abstracts  # noqa: E402
from colbert.utils.utils import load_colbert  # noqa: E402
from t5.label_prediction.inference import label_prediction  # noqa: E402
from t5.label_prediction.model import LP_MonoT5  # noqa: E402
from t5.sentence_selection.inference import rationale_selection  # noqa: E402
from t5.sentence_selection.model import SS_MonoT5  # noqa: E402

st.set_page_config(
    page_title="Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",  # CHANGE THESE
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

args = Namespace(
    query_maxlen=128,
    doc_maxlen=512,
    dim=128,
    similarity="l2",
    mask_punctuation=False,
    checkpoint="colbert/model_checkpoint/biobert-MM-2970159.pt",
)


# Main function
def main(args):
    db = start_connection("cord19_data/database/articles.sqlite")
    ar_model = load_ar_model(args)
    ss_model = load_ss_model()
    lp_model = load_lp_model()

    options = st.sidebar.radio(
        "Type of resulting articles to display",
        ["Supporting", "Contradicting", "Relevant"],
    )
    query = st.text_input(label="Enter a query or a claim.")

    rationales_selected, predicted_labels = search(query, ar_model, ss_model, lp_model)
    support, contradict, nei = categorize_results(predicted_labels)
    display_selection(options, db, support, contradict, nei, rationales_selected)
    close_connection(db)


# Connect to database
@st.cache(allow_output_mutation=True)
def start_connection(databse_url):
    db = sqlite3.connect(databse_url)
    return db


# Load abstract retrieval model
@st.cache
def load_ar_model(args):
    ar_model = load_colbert(args, do_log=True)
    return ar_model


# Load sentence selection model
@st.cache
def load_ss_model():
    ss_model = SS_MonoT5()
    return ss_model


# Load label prediction model
@st.cache
def load_lp_model():
    lp_model = LP_MonoT5()
    return lp_model


#
@st.cache
def search(query, ar_model, ss_model, lp_model):

    # Query input
    query_dict = {}
    query_dict[0] = query

    # Retrieve the most relevant document ids to the query or claim when users click on the 'Search' button
    if st.button("Search"):
        with st.spinner("Searching..."):

            abstracts_retrieved = retrieve_abstracts(query_dict, ar_model)

            rationales_selected = rationale_selection(
                query, abstracts_retrieved, ss_model
            )

            predicted_labels = label_prediction(query, rationales_selected, lp_model)
        st.success("Done!")
    return rationales_selected, predicted_labels


#
def categorize_results(predicted_labels):
    support, contradict, nei = [], [], []

    for id in predicted_labels:
        if predicted_labels[id] == "SUPPORT":
            support.append(id)
        elif predicted_labels[id] == "CONTRADICT":
            contradict.append(id)
        else:
            assert predicted_labels[id] == "NOT_ENOUGH_INFO"
            nei.append(id)

    return support, contradict, nei


#
def display_selection(options, db, support, contradict, nei, rationales_selected):
    cur = db.cursor()
    if options == "Supporting":
        display_results(cur, support, rationales_selected)
    elif options == "Contradicting":
        display_results(cur, contradict, rationales_selected)
    else:
        assert options == "Relevant"
        display_results(cur, nei, rationales_selected)


# Retrieve information about an article when provided with the abstract id
@st.cache
def retrieve_info(db, id):
    cur = db.cursor()

    abstract = cur.execute(
        "SELECT Abstract FROM articles WHERE Article_Id = (?)", (id,)
    ).fetchone()[0]
    title = cur.execute(
        "SELECT Title FROM articles WHERE Article_Id = (?)", (id,)
    ).fetchone()[0]
    published_date = cur.execute(
        "SELECT Published_Date FROM articles WHERE Article_Id = (?)", (id,)
    ).fetchone()[0]
    authors = cur.execute(
        "SELECT Authors FROM articles WHERE Article_Id = (?)", (id,)
    ).fetchone()[0]
    journal = cur.execute(
        "SELECT Journal FROM articles WHERE Article_Id = (?)", (id,)
    ).fetchone()[0]
    url = cur.execute(
        "SELECT Url FROM articles WHERE Article_Id = (?)", (id,)
    ).fetchone()[0]

    return abstract, title, published_date, authors, journal, url


def display_results(db, result_list, rationales_selected):
    cur = db.cursor()

    for i, abstract_id in enumerate(result_list):
        abstract, title, published_date, authors, journal, url = retrieve_info(
            cur, abstract_id
        )

        if published_date == "" or published_date is None:
            published_date = "Unknown published date"
        if authors == "" or authors is None:
            authors = "Uknown authors"
        if journal == "" or journal is None:
            journal = "Unknown source"

        st.subheader(f"{i}. [{title}]({url})")
        st.write(
            f"{authors}. <em>{journal}.</em> ({published_date})", unsafe_allow_html=True
        )

        sentence_indexes = rationales_selected[abstract_id]
        for idx in sentence_indexes:
            sentence = cur.execute(
                "SELECT Sentence FROM sentences WHERE Article_Id = (?) AND Sentence_Index = (?)",
                (abstract_id, idx),
            ).fetchone()[0]
            st.write(f"-  {sentence}")

        if st.checkbox("Show full abstract", key=i):
            st.write(f"{abstract}")


def close_connection(db):
    db.close()


# Run main()
if __name__ == "__main__":
    main(args)
