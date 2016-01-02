__author__ = 'bijoy'


class Sentiments:
    """
    Class containing list of sentiments and their Ids
    """
    SENTIMENTS = ["positive", "neutral", "negative"]

    @staticmethod
    def get_sentiment_id(sentiment):
        """
        Returns the id for an sentiment
        :param sentiment: the sentiment
        :return: the id
        """
        return Sentiments.SENTIMENTS.index(sentiment)

    @staticmethod
    def get_sentiment_for_id(id):
        """
        Returns the sentiment for the id
        :param id: the id
        :return: the sentiment
        """
        return Sentiments.SENTIMENTS[id]
