from common.base_classes import DocumentLevel

__author__ = 'bijoy'


class LastSentiment(DocumentLevel):
    """
    Returns the last sentiment for the sentences in the document
    """

    def get_prediction(self, tags=list(), sentences=None):
        if not tags or tags == []:
            return None
        return tags[-1]


class MostContinuousSentiment(DocumentLevel):
    """
    Returns the sentiment which occurred in the longest continuous sequence
    """

    def get_prediction(self, tags=list(), sentences=None):
        if not tags or tags == []:
            return None

        max_sentiment = self.get_most_continuous_tag(tags)
        return max_sentiment


class MostFrequentSentiment(DocumentLevel):
    """
    Returns the sentiment which occurred most frequently
    """

    def get_prediction(self, tags=list(), sentences=None):
        if not tags or tags == []:
            return None

        max_sentiment = self.get_max_tag(tags)
        return max_sentiment