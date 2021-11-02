from typing import Any, Mapping, Optional


class Query:
    def __init__(self, text: str, id: Optional[str] = None):
        self.text = text
        self.id = id


class Text:
    def __init__(
        self,
        text: str,
        metadata: Mapping[str, Any] = None,
        score: Optional[float] = 0,
        title: Optional[str] = None,
    ):
        self.text = text

        if metadata is None:
            metadata = dict()

        self.metadata = metadata
        self.score = score
        self.title = title
