from typing import List, Optional, Union, Mapping, Iterable
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch

from t5.base import Query, Text


TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int], List[List[int]], List[List[str]]]]


@dataclass
class QueryDocumentBatch:
    queries: List[Query]
    documents: List[Text]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.documents)


class TokenizerEncodeMixin:
    tokenizer: PreTrainedTokenizer = None 
    tokenizer_kwargs = None 

    def encode(self, strings: List[str]) -> TokenizerReturnType: 
        assert self.tokenizer and self.tokenizer_kwargs is not None, 'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings, **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings)) 
        return ret 


class QueryDocumentBatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                tokenizer: PreTrainedTokenizer,
                batch_size: int,
                model_type: str,
                pattern: str = '{query} {document}',
                **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs
        self.pattern = pattern

    def traverse_query_document(self, batch_input: QueryDocumentBatch) -> Iterable[QueryDocumentBatch]:
        for batch_idx in range(0, len(batch_input), self.batch_size): 
            queries = batch_input.queries[batch_idx : batch_idx + self.batch_size]
            docs = batch_input.documents[batch_idx : batch_idx + self.batch_size]
            assert len(queries) == len(docs)

            outputs = self.encode([self.pattern.format(query=queries[i].text, document=docs[i].text) for i in range(len(docs))])
            yield QueryDocumentBatch(queries, docs, outputs)


class T5BatchTokenizer(QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs): 
        if kwargs['model_type'] == 'sentence_selection':
            kwargs['pattern'] = 'Query: {query} Document: {document} Relevant:'
        else:
            assert kwargs['model_type'] == 'label_prediction'
            kwargs['pattern'] = 'hypothesis: {query} {document}' 

        if 'return_attention_mask' not in kwargs: 
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512

        super().__init__(*args, **kwargs)