import json
import os
import sqlite3
from collections import OrderedDict

from colbert.modeling.inference import ModelInference
from colbert.ranking.rankers import Ranker
from colbert.utils.utils import batch
from config.config import logger


def retrieve(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    db = sqlite3.connect("cord19_data/database/articles.sqlite")
    cur = db.cursor()

    faiss_part_range = ranker.faiss_index.faiss_part_range

    queries = args.queries
    qids_in_order = list(queries.keys())

    retrieved_abstracts = OrderedDict()

    for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
        for query_idx, qid in enumerate(qbatch):
            q_text = queries[qid]  # Query text

            Q = ranker.encode([q_text])
            pids, scores = ranker.rank(Q)

            # Find article ids corresponding to all of the passage ids retrieved
            all_results = OrderedDict()
            for i in range(len(pids)):
                pid, score = pids[i], scores[i]
                article_id = cur.execute(
                    "SELECT Article_Id FROM sections WHERE Section_Id = (?)", (pid,)
                ).fetchone()[0]

                if article_id not in all_results:
                    all_results[article_id] = {pid: score}
                else:
                    all_results[article_id][pid] = score

            # For each article id, find the passage id with the highest score
            final_results = OrderedDict()
            for article_id in all_results:
                max_aid_score = -100  # Max score for this article id

                for pid in all_results[article_id]:
                    if all_results[article_id][pid] > max_aid_score:
                        max_aid_score = all_results[article_id][
                            pid
                        ]  # Highest score for this article id
                        max_pid = pid  # pid with the highest score

                final_results[article_id] = (max_aid_score, max_pid)

            # Sort final_results by max_aid_score
            # sorted_results = [(article_id_1, (score1, pid1)), (article_id_2, (score2, pid2)), ...]
            sorted_results = sorted(
                final_results.items(), key=lambda x: x[1][0], reverse=True
            )

            # Retrieve abstracts with highest scores
            if len(sorted_results):

                top_results = sorted_results[: args.num_retrieved_abstracts]
                top_article_ids = [
                    top_results[i][0] for i in range(args.num_retrieved_abstracts)
                ]
                top_scores = [
                    top_results[i][1][0] for i in range(args.num_retrieved_abstracts)
                ]
                top_scores = [round(x, 2) for x in top_scores]

            # Save top results to file
            for i in range(len(top_results)):
                id = top_article_ids[i]
                score = top_scores[i]
                assert id not in retrieved_abstracts
                retrieved_abstracts[id] = score

            dir_path = os.path.join(args.index_path, "abstracts_retrieved")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if faiss_part_range is not None:
                fp = os.path.join(
                    dir_path,
                    f"abstracts_retrieved_{faiss_part_range.start}-{faiss_part_range.stop}.json",
                )
            else:
                fp = os.path.join(dir_path, "abstracts_retrieved.json")

            with open(fp, "w") as f:
                json.dump(retrieved_abstracts, f)

            # Log results
            logger.info(
                f"query_idx: {qoffset+query_idx}"
                f"qid: {qid}"
                f"query: {q_text}"
                f"len(sorted_results): {len(sorted_results)}"
                f"top article ids: {top_article_ids}"
                f"top scores: {top_scores}"
            )

            if faiss_part_range is not None:
                logger.info(
                    f"Finished retrieving abstracts using faiss indexes {faiss_part_range.start} - {faiss_part_range.stop}.\n"
                )
            else:
                logger.info("Finished retrieving abstracts.\n")
