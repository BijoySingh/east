from common.base_classes import DocumentLevel

__author__ = 'bijoy'

class LastEmotion(DocumentLevel):
    """
    Returns the last emotion for the sentences in the document
    """

    def get_prediction(self, tags=list(), sentences=None):
        if not tags or tags == []:
            return None
        return tags[-1]


class MostContinuousEmotion(DocumentLevel):
    """
    Returns the emotion which occurred in the longest continuous sequence
    """

    def get_prediction(self, tags=list(), sentences=None):
        if not tags or tags == []:
            return None

        max_emotion = self.get_most_continuous_tag(tags)
        return max_emotion

class MostFrequentEmotion(DocumentLevel):
    """
    Returns the emotion which occurred most frequently
    """

    def get_prediction(self, tags=list(), sentences=None):
        if not tags or tags == []:
            return None

        max_emotion = self.get_max_tag(tags)
        return max_emotion


