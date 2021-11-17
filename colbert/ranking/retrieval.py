import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(1, Path(__file__).parent.parent.parent.absolute().__str__())

from colbert.modeling.inference import ModelInference  # noqa: E402
from colbert.ranking.rankers import Ranker  # noqa: E402
from colbert.utils.utils import batch  # noqa: E402
from config.config import logger  # noqa: E402


def retrieve(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    db = sqlite3.connect("cord19_data/database/articles.sqlite")
    cur = db.cursor()

    milliseconds = 0

    queries = args.queries
    qids_in_order = list(queries.keys())

    abstracts_retrieved = OrderedDict()

    for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
        for query_idx, qid in enumerate(qbatch):
            q_text = queries[qid]  # Query text

            # torch.cuda.synchronize('cuda:0')
            s = time.time()

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
                max_aid_score = -1000000  # Max score for this article id

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

            # torch.cuda.synchronize()
            milliseconds += (time.time() - s) * 1000.0

            if len(sorted_results):

                highest_score = sorted_results[0][1][0]
                top_article_id = sorted_results[0][0]

                logger.info(
                    f"query_idx: {qoffset+query_idx}"
                    f"qid: {qid}"
                    f"query: {q_text}"
                    f"len(sorted_results): {len(sorted_results)}"
                    f"top article id: {top_article_id}"
                    f"highest score: {highest_score}"
                    f"retrieval time per query: {milliseconds / (qoffset+query_idx+1)} ms"
                )

                # Retrieve abstracts of the top 30 articles
                top_30_articles = sorted_results[:30]
                top_30_article_ids = [top_30_articles[i][0] for i in range(30)]

                # Record result
                assert qid not in abstracts_retrieved
                abstracts_retrieved[qid] = top_30_article_ids

    return abstracts_retrieved
