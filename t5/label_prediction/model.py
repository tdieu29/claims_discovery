from copy import deepcopy
from typing import List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

from t5.base import Query, Text
from t5.decode import greedy_decode
from t5.tokenize import QueryDocumentBatch, T5BatchTokenizer


class LP_MonoT5:
    def __init__(
        self,
        pretrained_model_name_or_path="t5/checkpoints/label_prediction/checkpoint-2646",
        use_amp=False,
        token_false="▁false",
        token_true="▁true",
        token_weak="▁weak",
    ):
        self.model = self.get_model(pretrained_model_name_or_path)
        self.tokenizer = self.get_tokenizer()
        (
            self.token_false_id,
            self.token_true_id,
            self.token_weak_id,
        ) = self.get_prediction_tokens(
            self.tokenizer, token_false, token_true, token_weak
        )
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(
        pretrained_model_name_or_path: str, *args, device: str = None, **kwargs
    ) -> T5ForConditionalGeneration:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        return (
            AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )
            .to(device)
            .eval()
        )

    @staticmethod
    def get_tokenizer(*args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:

        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained("t5-base", use_fast=False, *args, **kwargs),
            batch_size=batch_size,
            model_type="label_prediction",
        )

    @staticmethod
    def get_prediction_tokens(tokenizer, token_false, token_true, token_weak):
        token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
        token_true_id = tokenizer.tokenizer.get_vocab()[token_true]
        token_weak_id = tokenizer.tokenizer.get_vocab()[token_weak]

        return token_false_id, token_true_id, token_weak_id

    def rescore(self, queries: List[Query], texts: List[Text]) -> List[Text]:
        queries = deepcopy(queries)
        texts = deepcopy(texts)

        batch_input = QueryDocumentBatch(queries=queries, documents=texts)

        for batch in self.tokenizer.traverse_query_document(batch_input):
            batch_scores_list = []

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output["input_ids"].to(self.device)
                attn_mask = batch.output["attention_mask"].to(self.device)
                _, batch_scores = greedy_decode(
                    self.model,
                    input_ids,
                    length=1,
                    attention_mask=attn_mask,
                    return_last_logits=True,
                )

                batch_scores = batch_scores[
                    :, [self.token_false_id, self.token_true_id, self.token_weak_id]
                ]  # shape: (batch_size, 3)

                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)

                for i in range(batch_scores.shape[0]):
                    batch_scores_list.append(batch_scores[i, :].tolist())

            for doc, scores in zip(batch.documents, batch_scores_list):
                doc.score = scores

        return texts
