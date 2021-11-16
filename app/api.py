from argparse import Namespace
from datetime import datetime
from functools import wraps
from http import HTTPStatus

# import sqlite3
from typing import OrderedDict

from fastapi import FastAPI, Request

from app.schemas import RetrievePayload
from colbert.retrieve import retrieve_abstracts
from colbert.utils.utils import load_colbert
from t5.label_prediction.inference import label_prediction
from t5.label_prediction.model import LP_MonoT5
from t5.sentence_selection.inference import rationale_selection
from t5.sentence_selection.model import SS_MonoT5

app = FastAPI()

# Load models for abstract retrieval task, sentence selection task, label prediction task
arguments = Namespace()


# Load models to use for inference
@app.on_event("startup")
def app_startup_tasks():
    global ar_model
    global checkpoint
    global ss_model
    global lp_model
    global db
    global cur

    # Load models
    ar_model, checkpoint = load_colbert(arguments)

    ss_model = SS_MonoT5()
    ss_model.post_init()

    lp_model = LP_MonoT5()
    lp_model.post_init()

    # Connect to database
    # db = sqlite3.connect("cord19_data/database/articles.sqlite")
    # cur = db.cursor()


# @app.on_event("shutdown")
# def app_shutdown_tasks():
# Disconnect from database
# db.close()


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    """Health check."""
    results = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return results


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: RetrievePayload):
    """Retrieve and classify relevant document ids for a query or a claim."""

    query = payload.text
    abstracts = retrieve_abstracts(query, ar_model, checkpoint)
    rationales_selected = rationale_selection(query, abstracts, ss_model)
    predicted_labels = label_prediction(query, rationales_selected, lp_model)

    result = OrderedDict()
    for id in predicted_labels:
        result[id] = {
            "rationale_sentence_idx": rationales_selected[id],
            "label_prediction": predicted_labels[id],
        }

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": result,
    }

    return response
