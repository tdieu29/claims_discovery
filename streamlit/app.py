import sqlite3
import sys
from argparse import Namespace
from pathlib import Path
from typing import OrderedDict

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

    if query is not None:
        with st.spinner("Searching..."):
            rationales_selected, predicted_labels = search(
                query, ar_model, ss_model, lp_model
            )

        with st.spinner("Categorizing results..."):
            support, contradict, nei = categorize_results(predicted_labels)

        support_args = Namespace(
            db=db, results_list=support, rationales=rationales_selected
        )
        contradict_args = Namespace(
            db=db, results_list=contradict, rationales=rationales_selected
        )
        nei_args = Namespace(db=db, results_list=nei, rationales=rationales_selected)

        with st.spinner("Getting results..."):
            supporting_results = get_results(support_args)
            contradicting_results = get_results(contradict_args)
            nei_results = get_results(nei_args)

        if options == "Supporting":
            if len(supporting_results) > 0:
                display_results(supporting_results)
            else:
                st.write("No articles of this type found.")
        elif options == "Contradicting":
            if len(contradicting_results) > 0:
                display_results(contradicting_results)
            else:
                st.write("No articles of this type found.")
        else:
            assert options == "Relevant"
            if len(nei_results) > 0:
                display_results(nei_results)
            else:
                st.write("No articles of this type found.")


# Connect to database
@st.cache(hash_funcs={sqlite3.Connection: id})
def start_connection(databse_url):
    db = sqlite3.connect(databse_url, check_same_thread=False)
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


# Search database for relevant abstracts and rationale sentences in abstracts
# and predict labels
@st.cache
def search(query, ar_model, ss_model, lp_model):

    # Query input
    query_dict = {}
    query_dict[0] = query

    abstracts_retrieved = retrieve_abstracts(query_dict, ar_model)

    rationales_selected = rationale_selection(query, abstracts_retrieved, ss_model)

    predicted_labels = label_prediction(query, rationales_selected, lp_model)

    return rationales_selected, predicted_labels


# Categorize results into 3 lists: support, contradict, and NEI (not enough information)
@st.cache
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


# Function that checks arguments of the get_results function below
def check_get_results_args(args):
    db = args.db
    results_list = args.results_list
    rationales = args.rationales
    return (id(db), results_list, rationales)


@st.cache(hash_funcs={Namespace: check_get_results_args})
def get_results(args):
    db = args.db
    results_list = args.results_list
    rationales_selected = args.rationales

    cur = db.cursor()
    results_dictionary = OrderedDict()

    for abstract_id in results_list:

        retrieve_args = Namespace(db=db, abstract_id=abstract_id)

        # Retrieve information about the abstract with this abstract id
        abstract, title, published_date, authors, journal, url = retrieve_info(
            retrieve_args
        )

        if published_date == "" or published_date is None:
            published_date = "Unknown published date"
        if authors == "" or authors is None:
            authors = "Uknown authors"
        if journal == "" or journal is None:
            journal = "Unknown source"

        # Retrieve rationale sentences in this abstract
        rationale_sentences = []

        sentence_indexes = rationales_selected[abstract_id]
        for idx in sentence_indexes:
            sentence = cur.execute(
                "SELECT Sentence FROM sentences WHERE Article_Id = (?) AND Sentence_Index = (?)",
                (abstract_id, idx),
            ).fetchone()[0]

            rationale_sentences.append(sentence)

        results_dictionary[abstract_id] = [
            title,
            url,
            authors,
            journal,
            published_date,
            abstract,
            rationale_sentences,
        ]

    return results_dictionary


# Function that checks arguments of the retrieve_info function below
def check_retrieve_info_args(args):
    db = args.db
    abstract_id = args.abstract_id
    return (id(db), abstract_id)


# Retrieve information about an article when provided with the abstract id
@st.cache(hash_funcs={Namespace: check_retrieve_info_args})
def retrieve_info(retrieve_args):
    db = retrieve_args.db
    abstract_id = retrieve_args.abstract_id

    cur = db.cursor()

    abstract = cur.execute(
        "SELECT Abstract FROM articles WHERE Article_Id = (?)", (abstract_id,)
    ).fetchone()[0]
    title = cur.execute(
        "SELECT Title FROM articles WHERE Article_Id = (?)", (abstract_id,)
    ).fetchone()[0]
    published_date = cur.execute(
        "SELECT Published_Date FROM articles WHERE Article_Id = (?)", (abstract_id,)
    ).fetchone()[0]
    authors = cur.execute(
        "SELECT Authors FROM articles WHERE Article_Id = (?)", (abstract_id,)
    ).fetchone()[0]
    journal = cur.execute(
        "SELECT Journal FROM articles WHERE Article_Id = (?)", (abstract_id,)
    ).fetchone()[0]
    url = cur.execute(
        "SELECT Url FROM articles WHERE Article_Id = (?)", (abstract_id,)
    ).fetchone()[0]

    return abstract, title, published_date, authors, journal, url


def display_results(results_dictionary):
    count = 0

    for abstract_id in results_dictionary:
        (
            title,
            url,
            authors,
            journal,
            published_date,
            abstract,
            rationale_sentences,
        ) = results_dictionary[abstract_id]

        st.subheader(f"{count}. [{title}]({url})")
        st.write(
            f"{authors}. <em>{journal}.</em> ({published_date})", unsafe_allow_html=True
        )

        for sentence in rationale_sentences:
            st.write(f"-  {sentence}")

        if st.checkbox("Show full abstract", key=abstract_id):
            st.write(f"{abstract}")

        count += 1


# Run main()
if __name__ == "__main__":
    main(args)
