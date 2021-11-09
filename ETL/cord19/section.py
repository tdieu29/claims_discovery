"""
Section module
"""

import re

import nltk

from ..text import Text


class Section:
    @staticmethod
    def parse(row):
        """
        Reads title and abstract for a given row. Text is returned as a list of text spans.

        Args:
            row ([type]): input row
            directory ([type]): input directory

        Returns:
            list of text spans
        """

        sections = []

        # Return empty list if there is no title and abstract in row
        if not row["title"] and not row["abstract"]:
            return sections

        # Add a period at the end of title if necessary
        if row["title"][-1].isalnum():
            text = row["title"] + ". " + row["abstract"]
        else:
            text = row["title"] + " " + row["abstract"]

        if text:
            assert type(text) == str

            # Remomve leading and trailing []
            text = re.sub(r"^\[", "", text)
            text = re.sub(r"\]$", "", text)

            # Transform and clean text
            text = Text.transform(text)

            # Segment each document ("text") into spans by applying a sliding window of 6 sentences with a stride of 3
            spans = Section.get_spans(text)
            for i in range(len(spans)):
                sections.append((i, spans[i]))

        return sections

    @staticmethod
    def get_spans(text):
        num_sent = 6  # Number of sentences in each span
        stride = 3

        # Get individual sentences in each document
        sentences = nltk.sent_tokenize(text)

        # Segment each document into spans of text
        spans = []
        i = 0
        if len(sentences) <= 1:
            spans.append(sentences)
        else:
            while i < len(sentences) - 1:
                spans.append(sentences[i : i + num_sent])
                i += stride

        # Join individual sentences in each span into one paragraph
        new_spans = []
        for item in spans:
            new_item = " ".join(item)
            new_spans.append(new_item)

        # Make sure the number of spans stays the same
        assert len(spans) == len(new_spans)

        return new_spans
