from collections import OrderedDict
import time
import json
import sqlite3

from colbert.modeling.inference import ModelInference
from colbert.ranking.rankers import Ranker
from colbert.utils.utils import batch


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
            q_text = queries[qid] # Query text

            #torch.cuda.synchronize('cuda:0')
            s = time.time()

            Q = ranker.encode([q_text])
            pids, scores = ranker.rank(Q)
                
            # Find article ids corresponding to all of the passage ids retrieved 
            all_results = OrderedDict()
            for i in range(len(pids)):
                pid, score = pids[i], scores[i]
                article_id = cur.execute("SELECT Article_Id FROM sections WHERE Section_Id = (?)", (pid,)).fetchone()[0] 
                    
                if article_id not in all_results:
                    all_results[article_id] = {pid: score}
                else:
                    all_results[article_id][pid] = score
                    
            # For each article id, find the passage id with the highest score
            final_results = OrderedDict()
            for article_id in all_results:
                max_aid_score = -1000000 # Max score for this article id 
                    
                for pid in all_results[article_id]: 
                    if all_results[article_id][pid] > max_aid_score:
                        max_aid_score = all_results[article_id][pid] # Highest score for this article id 
                        max_pid = pid # pid with the highest score 
                    
                final_results[article_id] = (max_aid_score, max_pid)

            # Sort final_results by max_aid_score
            # sorted_results = [(article_id_1, (score1, pid1)), (article_id_2, (score2, pid2)), ...]
            sorted_results = sorted(final_results.items(), key=lambda x: x[1][0], reverse=True)

            #torch.cuda.synchronize()
            milliseconds += (time.time() - s) * 1000.0

            if len(sorted_results):
                               
                highest_score = sorted_results[0][1][0]
                top_article_id = sorted_results[0][0]
                
                #print(qoffset+query_idx, q, len(sorted_results), highest_score, top_article_id,
                #        milliseconds / (qoffset+query_idx+1), 'ms')
            
                # Retrieve abstracts of the top 50 articles 
                top_30_articles = sorted_results[:30]
                top_30_article_ids = [top_30_articles[i][0] for i in range(30)]
                
                # Record results 
                assert qid not in abstracts_retrieved
                abstracts_retrieved[qid] = top_30_article_ids 
                                       
    print('\n\n')
    print("#> Done.")
    print('\n\n')

    return abstracts_retrieved

