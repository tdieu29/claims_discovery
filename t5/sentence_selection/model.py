from typing import List
from transformers import T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy
import torch

from t5.base import Query, Text
from t5.decode import greedy_decode
from t5.tokenize import T5BatchTokenizer, QueryDocumentBatch


class SS_MonoT5(): 
  def __init__(self,
               pretrained_model_name_or_path = '',              
               use_amp = False,
               #token_false = '▁false', 
               #token_true = '▁true'
               ):
    #self.model = self.get_model(pretrained_model_name_or_path)
    #self.tokenizer = self.get_tokenizer()
    #self.token_false_id, self.token_true_id = self.get_prediction_tokens(self.tokenizer,
    #                                                                     token_false,
    #                                                                     token_true)
    self.model = None
    self.tokenizer = None
    self.token_false_id, self.token_true_id = None
    self.device = None

    self.pretrained_model_name_or_path = pretrained_model_name_or_path 
    #self.device = next(self.model.parameters(), None).device
    self.use_amp = use_amp 

  def post_init(self, token_false = '▁false', token_true = '▁true'):
    self.model = self.get_model(self.pretrained_model_name_or_path)
    self.tokenizer = self.get_tokenizer()
    self.token_false_id, self.token_true_id = self.get_prediction_tokens(self.tokenizer,
                                                                         token_false,
                                                                         token_true)
    self.device = next(self.model.parameters(), None).device
    return None

  @staticmethod
  def get_model(pretrained_model_name_or_path: str, *args, device: str = None, **kwargs) -> T5ForConditionalGeneration: 
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    return AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs).to(device).eval()

  @staticmethod 
  def get_tokenizer(*args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
    return T5BatchTokenizer(AutoTokenizer.from_pretrained("t5-base", use_fast=False, *args, **kwargs),
                            batch_size=batch_size, 
                            model_type='sentence_selection')
              
  @staticmethod 
  def get_prediction_tokens(tokenizer, token_false, token_true):
    token_false_id = tokenizer.tokenizer.get_vocab()[token_false] 
    token_true_id  = tokenizer.tokenizer.get_vocab()[token_true]
    return token_false_id, token_true_id

  def rescore(self, queries: List[Query], texts: List[Text]) -> List[Text]:
    queries = deepcopy(queries)
    texts = deepcopy(texts)
   
    batch_input = QueryDocumentBatch(queries=queries, documents=texts)
    
    for batch in self.tokenizer.traverse_query_document(batch_input):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            input_ids = batch.output['input_ids'].to(self.device)
            attn_mask = batch.output['attention_mask'].to(self.device) 
            _, batch_scores = greedy_decode(self.model,
                                        input_ids,
                                        length=1,
                                        attention_mask=attn_mask,
                                        return_last_logits=True)
        
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]] # shape: (batch_size, 2)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    
            batch_log_probs = batch_scores[:, 1].tolist()

        for doc, score in zip(batch.documents, batch_log_probs): 
            doc.score = score
    
    return texts 