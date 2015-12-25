from progressbar import ProgressBar
from .data import SentiWordNet, OpinionLexicon
from common.base_classes import SentenceLevel
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from .sentiments import Sentiments
from utilities.storage import Storage

__author__ = 'bijoy'


class MaxSentimentScore(SentenceLevel):
    """Returns the Sentiment with the maximum score as per the sentiwordnet"""

    def __init__(self, allow_negation=False):
        SentenceLevel.__init__(self)
        self.word_sentiment_scores = SentiWordNet(allow_negation=allow_negation)

    def get_prediction(self, sentence):
        # Split into words
        words = self.text_utility.tokenize(sentence.lower())

        sentiment_scores = self.word_sentiment_scores.get_sentiment_scores(words)

        if sentiment_scores[1] >= sentiment_scores[0]:
            return 'positive'

        return 'negative'


class OpinionLexiconSentimentCount(SentenceLevel):
    """Returns the Sentiment with the maximum sentiment as per the opinion lexicon"""

    def __init__(self, allow_negation=False):
        SentenceLevel.__init__(self)
        self.opinion_lexicon = OpinionLexicon(allow_negation=allow_negation)

    def get_prediction(self, sentence):
        # Split into words
        words = self.text_utility.tokenize(sentence.lower())
        sentiment_counts = self.opinion_lexicon.get_opinion_count(words)

        if sentiment_counts[1] >= sentiment_counts[0]:
            return 'positive'

        return 'negative'


class WordSentimentClassifier(SentenceLevel):
    """
    Classify using the feature set of unigrams in the document
    """
    FOLDER = 'pickled/sentiment/'
    FILENAME = 'classifier_word_model'
    WORD_LIMIT = 20000

    def __init__(self, allow_negation=True, filename=FILENAME):
        SentenceLevel.__init__(self, filename)
        self.allow_negation = allow_negation
        self.engine = None
        self.trained = False
        self.word_set = dict()

    def create_word_vector(self, words):
        count_list = [0 for i in range(len(self.word_set))]
        for word in words:
            if word in self.word_set.keys():
                count_list[self.word_set[word]] = 1
        return count_list

    def get_normalized_words(self, line):
        words = self.text_utility.tokenize(line)
        if self.allow_negation:
            words = self.text_utility.negate_words(words)
        words = self.text_utility.remove_punctuations(words)
        words = self.text_utility.lemmatize(words)
        return words

    def train_words(self, training_set):
        bar = ProgressBar()
        word_count = defaultdict(lambda: 0)
        for tagged_line in bar(training_set):
            line = tagged_line[0]
            words = self.get_normalized_words(line.lower())
            for word in words:
                word_count[word] += 1

        keys = sorted(word_count.items(),
                      key=word_count.get,
                      reverse=True)[:min(self.WORD_LIMIT, len(word_count))]
        self.word_set.clear()
        for word, frequency in keys:
            self.word_set[word] = len(self.word_set)

    def train(self, training_set, save_file=True):
        classifier_x = []
        classifier_y = []

        self.train_words(training_set)

        bar = ProgressBar()
        for tagged_line in bar(training_set):
            line = tagged_line[0]
            sentiment = tagged_line[1]
            words = self.get_normalized_words(line)
            x = self.create_word_vector(words)

            classifier_x.append(x)
            classifier_y.append(Sentiments.get_sentiment_id(sentiment))

        self.engine.fit(classifier_x, classifier_y)
        self.trained = True

        if save_file:
            Storage.dump(self.filename, (self.engine, self.word_set))

    def get_prediction(self, sentence):
        # Split into words
        if not self.trained:
            engine = Storage.load(self.filename)
            if engine:
                self.engine = engine[0]
                self.word_set = engine[1]
                self.trained = True

        words = self.get_normalized_words(sentence.lower())
        sentiment_id = self.engine.predict([self.create_word_vector(words)])[0]

        return Sentiments.get_sentiment_for_id(sentiment_id)


class UnigramSentimentSVM(WordSentimentClassifier):
    """Classify on a support vector machine using the feature set of unigrams in the document"""

    FILENAME = 'svm_word_model'
    WORD_LIMIT = 15000

    def __init__(self, allow_negation=True, filename=FILENAME):
        WordSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = SVC()


class UnigramSentimentGaussianNB(WordSentimentClassifier):
    """Classify on a gaussian naive bayes classifier using the feature set of unigrams in the document"""

    FILENAME = 'gaussian_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        WordSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = GaussianNB()


class UnigramSentimentBernoulliNB(WordSentimentClassifier):
    """Classify on a bernoulli naive bayes classifier using the feature set of unigrams in the document"""

    FILENAME = 'bernoulli_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        WordSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = BernoulliNB()


class UnigramSentimentMultinomialNB(WordSentimentClassifier):
    """Classify on a multinomial naive bayes classifier using the feature set of unigrams in the document"""

    FILENAME = 'multinomial_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        WordSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = MultinomialNB()


class UnigramSentimentMaxEnt(WordSentimentClassifier):
    """Classify on a maximum entropy classifier using the feature set of unigrams in the document"""

    FILENAME = 'max_ent_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        WordSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = LogisticRegression()


class BigramSentimentClassifier(WordSentimentClassifier):
    """
    Classify using the feature set of bigrams in the document
    """

    FOLDER = 'pickled/sentiment/'
    FILENAME = 'bigram_word_model'

    def get_normalized_words(self, line):
        words = self.text_utility.tokenize(line)
        if self.allow_negation:
            words = self.text_utility.negate_words(words)
        words = self.text_utility.remove_punctuations(words)

        bigrams = list()
        previous_word = '$'
        for word in words:
            bigrams.append((previous_word, word))
            previous_word = word
        bigrams.append((words[-1], '^'))

        return bigrams


class BigramSentimentMultinomialNB(BigramSentimentClassifier):
    """Classify on a multinomial naive bayes model using the feature set of bigrams in the document"""

    FILENAME = 'bigram_multinomial_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        BigramSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = MultinomialNB()


class BigramSentimentGaussianNB(BigramSentimentClassifier):
    """Classify on a gaussian naive bayes model using the feature set of bigrams in the document"""

    FILENAME = 'bigram_gaussian_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        BigramSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = GaussianNB()


class BigramSentimentBernoulliNB(BigramSentimentClassifier):
    """Classify on a bernoulli naive bayes model using the feature set of bigrams in the document"""

    FILENAME = 'bigram_bernoulli_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        BigramSentimentClassifier.__init__(self, allow_negation, filename)
        self.engine = BernoulliNB()
