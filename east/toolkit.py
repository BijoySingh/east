from east.emotion_analysis.sentence_level import *
from east.emotion_analysis.document_level import *
from east.sentiment_analysis.sentence_level import *
from east.sentiment_analysis.document_level import *

from east.utilities.text import Text

__author__ = 'bijoy'


class Toolkit:
    """
    API class, acts as a quick interface for integration into other python applications
    """
    sl_sentiment = [UnigramSentimentMultinomialNB, UnigramSentimentBernoulliNB,
                    UnigramSentimentMaxEnt, UnigramSentimentGaussianNB,
                    OpinionLexiconSentimentCount, BigramSentimentMultinomialNB,
                    BigramSentimentBernoulliNB, BigramSentimentGaussianNB,
                    UnigramSentimentSVM, MaxSentimentScore]
    dl_sentiment = [MostFrequentSentiment, LastSentiment, MostContinuousSentiment]
    sl_emotion = [EmotionScoreMultinomialNB, EmotionScoreSVM, UnigramEmotionMultinomialNB,
                  UnigramEmotionBernoulliNB, EmotionScoreGaussianNB, UnigramEmotionGaussianNB,
                  MaxEmotionScore, UnigramEmotionSVM, EmotionScoreBernoulliNB]
    dl_emotion = [MostFrequentEmotion, LastEmotion, MostContinuousEmotion]

    def __init__(self, sentiment=False, sentence_level=0, document_level=0, club=False):
        """
        Constructor for the API
        :param sentiment: is sentiment analysis
        :param sentence_level: sentence level algorithm id
        :param document_level: document level algorithm id
        :param club: want to club entire document as one sentence
        :return:
        """
        self.sentiment = sentiment
        if self.sentiment:
            self.sentence_level = self.get_sentence_level_sentiment_algorithm(sentence_level)
            self.document_level = self.get_document_level_sentiment_algorithm(document_level)
        else:
            self.sentence_level = self.get_sentence_level_emotion_algorithm(sentence_level)
            self.document_level = self.get_document_level_emotion_algorithm(document_level)
        self.club = club

    @staticmethod
    def get_algorithm(algorithms, algorithm_id):
        """
        From the list of algorithm classes provided, it tries to return an instance of the algorithm at algorithm_id
        :param algorithms: the list of algorithm classes
        :param algorithm: the position of the algorithm class in the list
        :return: instance of the algorithm or falls back to the first algorithm in the list
        """
        if algorithm_id < len(algorithms):
            return algorithms[algorithm_id]()
        return algorithms[0]()

    def get_sentence_level_emotion_algorithm(self, algorithm_id):
        """
        Returns an instance of a sentence level emotion analysis algorithm with the given algorithm id
        :param algorithm_id: the position of the algorithm class in the list
        :return: instance of the algorithm or falls back to the first algorithm in the list
        """
        return self.get_algorithm(self.sl_emotion, algorithm_id)

    def get_sentence_level_sentiment_algorithm(self, algorithm_id):
        """
        Returns an instance of a sentence level sentiment analysis algorithm with the given algorithm id
        :param algorithm_id: the position of the algorithm class in the list
        :return: instance of the algorithm or falls back to the first algorithm in the list
        """
        return self.get_algorithm(self.sl_sentiment, algorithm_id)

    def get_document_level_emotion_algorithm(self, algorithm_id):
        """
        Returns an instance of a document level emotion analysis algorithm with the given algorithm id
        :param algorithm_id: the position of the algorithm class in the list
        :return: instance of the algorithm or falls back to the first algorithm in the list
        """
        return self.get_algorithm(self.dl_emotion, algorithm_id)

    def get_document_level_sentiment_algorithm(self, algorithm_id):
        """
        Returns an instance of a document level sentiment analysis algorithm with the given algorithm id
        :param algorithm_id: the position of the algorithm class in the list
        :return: instance of the algorithm or falls back to the first algorithm in the list
        """
        return self.get_algorithm(self.dl_sentiment, algorithm_id)

    def add_sentence_level_emotion_algorithm(self, algorithm_class):
        """
        Add a new sentence level emotion algorithm
        :param algorithm_class: the class of the algorithm (must extend or implement functions of SentenceLevel)
        :return: index of this algorithm
        """
        self.sl_emotion.append(algorithm_class)
        return len(self.sl_emotion) - 1

    def add_sentence_level_sentiment_algorithm(self, algorithm_class):
        """
        Add a new sentence level sentiment algorithm
        :param algorithm_class: the class of the algorithm (must extend or implement functions of SentenceLevel)
        :return: index of this algorithm
        """
        self.sl_sentiment.append(algorithm_class)
        return len(self.sl_sentiment) - 1

    def add_document_level_emotion_algorithm(self, algorithm_class):
        """
        Add a new document level emotion algorithm
        :param algorithm_class: the class of the algorithm (must extend or implement functions of DocumentLevel)
        :return: index of this algorithm
        """
        self.dl_emotion.append(algorithm_class)
        return len(self.dl_emotion) - 1

    def add_document_level_sentiment_algorithm(self, algorithm_class):
        """
        Add a new document level sentiment algorithm
        :param algorithm_class: the class of the algorithm (must extend or implement functions of DocumentLevel)
        :return: index of this algorithm
        """
        self.dl_sentiment.append(algorithm_class)
        return len(self.dl_sentiment) - 1


    def analyse(self, document):
        """
        :param document: The text that needs to be emotion / sentiment analysed
        :return: a dict object containing the emotion / sentiment of the document
        as well all the sentences of the document
        """
        sentences = [document] if self.club else Text().split_document(document)

        tags = list()
        for sentence in sentences:
            tags.append(self.sentence_level.get_prediction(sentence))
        tag = self.document_level.get_prediction(tags=tags)
        return {"tag": tag, "tags": tags}

    @staticmethod
    def get_help_string(list, delimiter=", "):
        """
        Generates the help string for the command line interface and otherwise
        :param list: the list of algorithm classes
        :param delimiter: a delimiter between 2 list items
        :return:
        """
        list_string = ""
        for position in range(len(list)):
            list_string += str(position) + ' => ' + str(list[position].__name__) + delimiter
        return list_string

    @staticmethod
    def get_help(emotion=True, sentence=True):
        """
        Returns the help for a particular kind of algorithm request
        :param emotion: is for emotion analysis
        :param sentence: is sentence level
        :return: help string
        """
        if emotion and sentence:
            return "Emotion Analysis: " + Toolkit.get_help_string(Toolkit.sl_emotion)
        elif emotion and not sentence:
            return "Emotion Analysis: " + Toolkit.get_help_string(Toolkit.dl_emotion)
        elif not emotion and sentence:
            return "Sentiment Analysis: " + Toolkit.get_help_string(Toolkit.sl_sentiment)
        elif not emotion and not sentence:
            return "Sentiment Analysis: " + Toolkit.get_help_string(Toolkit.dl_sentiment)
