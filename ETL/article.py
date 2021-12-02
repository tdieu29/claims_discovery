"""
Article module.
"""


class Article:
    """
    Article object. Holds metadata, abstract and title of that article.
    """

    def __init__(self, metadata, sections, sentences):
        """
        Stores article metadata and content as an object.

        Args:
            metadata ([tuple]): article metadata
            sections ([list]): spans of text from the abstract of the article
        """

        self.metadata = metadata
        self.sections = sections
        self.sentences = sentences

    def uid(self):
        """
        Returns the article uid.

        Returns:
            article uid
        """

        return self.metadata[0]
