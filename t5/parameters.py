# Sentence prediction training parameters
def ss_params():
    triples_path = [
        "--triples_path",
        "/content/drive/MyDrive/NLP_Final_Project/Sentence_Selection/scifact_ss_train.txt",
        "/content/drive/MyDrive/NLP_Final_Project/Sentence_Selection/bio_claim_ss_train.txt",
        "/content/drive/MyDrive/NLP_Final_Project/Sentence_Selection/bio_query_ss_train.txt",
    ]
    output_model_path = [
        "--output_model_path",
        "/content/drive/MyDrive/NLP_Final_Project/Sentence_Selection/3e-4_4epochs",
    ]
    final_list = triples_path + output_model_path
    return final_list


# Label prediction training parameters
def lp_params():
    triples_path = [
        "--triples_path",
        "/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/scifact_lp_train.txt",
        "/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/bio_claim_lp_train.txt",
        "/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/bio_query_lp_train.txt",
    ]
    output_model_path = [
        "--output_model_path",
        "/content/drive/MyDrive/NLP_Final_Project/Label_Prediction/3e-4_6epochs",
    ]
    final_list = triples_path + output_model_path
    return final_list
