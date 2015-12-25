import re
import logging

has_nltk_tokenizer = True
try:
    from nltk.tokenize import TweetTokenizer
except ImportError:
    logging.warning("nltk TweetTokenizer not found, please install / download to improve performance")
    has_nltk_tokenizer = False

has_nltk_lemmatizer = True
try:
    from nltk.stem.wordnet import WordNetLemmatizer
except ImportError:
    logging.warning("nltk WordNetLemmatizer not found, please install / download to improve performance")
    has_nltk_lemmatizer = False

has_nltk_data = True
try:
    import nltk.data
except ImportError:
    logging.warning("nltk.data not found, please install / download to improve performance")
    has_nltk_data = False

__author__ = 'bijoy'


class Text:
    """Some basic text processing functions"""
    NEGATIVE_WORDS = ['not', 'but', 'although', 'though', 'don\'t', 'dont', 'didnt', 'didn\'t', 'n\'t', 'isnt',
                      'isn\'t']
    STOP_WORDS = ['the', 'as', 'like', 'is', 'a', 'an', 'in', 'on', 'of']

    def __init__(self):
        if has_nltk_tokenizer:
            self.tokenizer = TweetTokenizer()
        if has_nltk_lemmatizer:
            self.lemmatizer = WordNetLemmatizer()
        if has_nltk_data:
            self.sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def unicode_to_ascii(sentence):
        """
        Converts unicode to ascii for a sentence
        :param sentence:
        :return:
        """
        return sentence.decode('unicode_escape').encode('ascii', 'ignore')

    def tokenize(self, sentence):
        """
        Tokenize the sentence to the composing words
        :param sentence: the sentence
        :return: the list of words
        """
        sentence = self.unicode_to_ascii(sentence)

        if has_nltk_tokenizer:
            return map(str, self.tokenizer.tokenize(sentence))
        return sentence.split()

    @staticmethod
    def is_punctuation(word):
        """
        Returns true if the word is a punctuation
        :param word: the word
        :return: is the word a punctuation (True / False)
        """
        return not re.search('[a-zA-Z0-9]', word)

    def split_document(self, document):
        """
        Splits the document to a list of sentences
        :param document: the document
        :return: a list of sentences
        """
        document = self.unicode_to_ascii(document)
        if has_nltk_data:
            return self.sentence_detector.tokenize(document)
        return re.split(r' *[\.\?!][\'"\)\]]* *', document)

    @staticmethod
    def remove_stop_words(words):
        """
        Removes stop words from a list of words
        :param words:  the list of words
        :return: the filtered list
        """
        return [word for word in words if word.lower() not in Text.STOP_WORDS]

    @staticmethod
    def remove_punctuations(words):
        """
        Removes punctuations from the words
        :param words: the list of words
        :return: the filtered list of words
        """
        return [word for word in words if not Text.is_punctuation(word)]

    @staticmethod
    def negate_words(words):
        """
        Negates words in a list of words
        :param words: list of words
        :return: a modified list of words (adds NOT_ for negated words)
        """
        negation_started = False
        n_words = []
        for word in words:
            if negation_started:
                n_words.append('NOT_' + word)
            else:
                n_words.append(word)

            if word in Text.NEGATIVE_WORDS:
                negation_started = not negation_started
            if Text.is_punctuation(word):
                negation_started = False
        return n_words

    def lemmatize(self, words):
        """
        Lemmatize words in a list of words
        :param words: the list of words
        :return: the lemmatized word list
        """
        if has_nltk_lemmatizer:
            return [self.lemmatizer.lemmatize(word) for word in words]
        return words
