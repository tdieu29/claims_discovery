import math
import sqlite3
from typing import OrderedDict

from nltk import sent_tokenize

from t5.base import Query, Text


def rationale_selection(query, abstracts_retrieved, SS_MonoT5_model):

    top_50_article_ids = list(abstracts_retrieved.keys())[0]

    # Retrieve abstracts and create inputs for the Sentence Selection model
    db = sqlite3.connect("cord19_data/database/articles.sqlite")
    cur = db.cursor()

    all_sentences = OrderedDict()
    all_queries, all_documents = [], []

    for id in top_50_article_ids:
        abstract_text = cur.execute(
            "SELECT Abstract FROM articles WHERE Article_Id = (?)", (id,)
        ).fetchone()[0]
        sentences = sent_tokenize(abstract_text)

        for i in range(len(sentences)):
            q = Query(text=query.strip())
            doc = Text(text=sentences[i].strip())

            all_queries.append(q)
            all_documents.append(doc)

            key = str(id) + "#" + str(i)
            all_sentences[key] = {"score": None, "text": sentences[i].strip()}

    # Score each sentence in each abstract
    SS_scored_documents = SS_MonoT5_model.rescore(
        queries=all_queries, texts=all_documents
    )

    # Add the score of each sentence to the all_sentences dictionary
    keys = list(all_sentences.keys())

    for i in range(len(SS_scored_documents)):
        assert (
            SS_scored_documents[i].text
            == all_documents[i].text
            == all_sentences[keys[i]]["text"]
        )
        assert SS_scored_documents[i].score is None
        all_sentences[keys[i]]["score"] = SS_scored_documents[i].score

    # Select rationale sentences in each abstract
    rationales_selected = OrderedDict()

    for key in all_sentences:
        abstract_id, sent_idx = key.split("#")
        abstract_id = int(abstract_id)
        sent_idx = int(sent_idx)

        if math.exp(all_sentences[key]["score"]) >= 0.999:
            rationales_selected[abstract_id] = rationales_selected.get(abstract_id, [])
            rationales_selected[abstract_id].append(sent_idx)

    assert len(rationales_selected) == len(top_50_article_ids)

    db.close()

    return rationales_selected
