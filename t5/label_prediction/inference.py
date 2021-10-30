import sqlite3
import math
from typing import OrderedDict
from nltk import sent_tokenize
from t5.base import Query, Text 


def label_prediction(query, rationales_selected, LP_MonoT5_model):

    label_map = {0: 'CONTRADICT', 1: 'SUPPORT', 2: 'NOT_ENOUGH_INFO'}
    
    # Connect to database 
    db = sqlite3.connect("cord19_data/database/articles.sqlite")
    cur = db.cursor()

    # Create inputs for the Label Prediction model
    all_queries = []
    all_documents = []

    for abstract_id in rationales_selected:
        sent_indexes = sorted(rationales_selected[abstract_id]) # Indexes of rationale sentences in abstract
    
        # Retrieve all sentences in abstract
        abstract_text = cur.execute("SELECT Abstract FROM articles WHERE Article_Id = (?)", (abstract_id,)).fetchone()[0] 
        sentences = sent_tokenize(abstract_text) 
        
        # Doc
        evidence = ' '.join(["sentence{}: ".format(idx+1) + sentences[index].strip() for idx, index in enumerate(sent_indexes)])
        doc = Text(text=evidence)
        all_documents.append(doc)
        
        # Query
        q = Query(text=query.strip()) 
        all_queries.append(q)

    # Score each abstract
    LP_scored_documents = LP_MonoT5_model.rescore(queries=all_queries, texts=all_documents)
    
    # Predict label of each abstract
    doc_ids = list(rationales_selected.keys())
    assert len(doc_ids) == len(LP_scored_documents)  # DELETE LATER?
    
    predicted_labels = OrderedDict()
    for i in range(len(LP_scored_documents)):

        # Verify that we got the correct abstract
        assert LP_scored_documents[i].text == all_documents[i].text

        abstract = cur.execute("SELECT Abstract FROM articles WHERE Article_Id = (?)", (doc_ids[i],)).fetchone()[0] 
        first_rationale_sent_idx = rationales_selected[doc_ids[i]][0]
        first_rationale_sent = sent_tokenize(abstract)[first_rationale_sent_idx].strip()
        length = len(first_rationale_sent) # Length of the first rationale sentence in this abstract
        assert LP_scored_documents[i].text[11:length] in abstract

        # Determine label of this abstract
        all_scores = LP_scored_documents[i].score
        probabilities = [math.exp(score) for score in all_scores]
        max_prob = max(probabilities)
        max_prob_idx = probabilities.index(max_prob)
        label = label_map[max_prob_idx]

        # Record the predicted label for this abstract
        predicted_labels[doc_ids[i]] = label

    db.close()

    return predicted_labels
