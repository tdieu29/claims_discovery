"""
Sentence module
"""
import re

import nltk

from ..text import Text


class Sentence:
    @staticmethod
    def parse(row):
        sentence_list = []

        abstract = row["abstract"]
        if not abstract:
            return sentence_list

        # Remomve leading and trailing []
        assert type(abstract) == str
        abstract = re.sub(r"^\[", "", abstract)
        abstract = re.sub(r"\]$", "", abstract)

        # Transform and clean text
        abstract = Text.transform(abstract)

        # Get individual sentences in abstract
        sentences = nltk.sent_tokenize(abstract)
        for i in range(len(sentences)):
            sentence_list.append((i, sentences[i]))

        return sentence_list
