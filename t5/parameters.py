# Sentence prediction training parameters
def ss_params():
    triples_path = [
        "--triples_path",
        "t5/train_data/sentence_selection/scifact_ss_train.txt",
        "t5/train_data/sentence_selection/bio_claim_ss_train.txt",
        "t5/train_data/sentence_selection/bio_query_ss_train.txt",
    ]
    output_model_path = [
        "--output_model_path",
        "t5/checkpoints/sentence_selection",
    ]
    final_list = triples_path + output_model_path
    return final_list


# Label prediction training parameters
def lp_params():
    triples_path = [
        "--triples_path",
        "t5/train_data/label_prediction/scifact_lp_train.txt",
        "t5/train_data/label_prediction/bio_claim_lp_train.txt",
        "t5/train_data/label_prediction/bio_query_lp_train.txt",
    ]
    output_model_path = [
        "--output_model_path",
        "t5/checkpoints/label_prediction",
    ]
    final_list = triples_path + output_model_path
    return final_list
