import random

from progressbar import ProgressBar
from .data import WordEmotionScore
from .emotions import Emotions
from east.common.base_classes import SentenceLevel
from east.utilities.storage import Storage
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

__author__ = 'bijoy'

class EmotionScoreClassifier(SentenceLevel):
    """Trains a classifier based on the Emotion Scores of the Sentence"""

    FOLDER = 'pickled/emotion/'
    FILENAME = 'classifier_emotion_model'

    def __init__(self, allow_negation=False, filename=FILENAME):
        SentenceLevel.__init__(self, filename)
        self.word_emotion_scores = WordEmotionScore(allow_negation=allow_negation)
        self.engine = None
        self.trained = False

    def train(self, training_set, save_file=True):
        classifier_x = []
        classifier_y = []

        bar = ProgressBar()
        for tagged_line in bar(training_set):
            line = tagged_line[0]
            emotion = tagged_line[1]

            words = self.text_utility.tokenize(line)
            emotion_scores = self.word_emotion_scores.get_emotion_scores(words)
            x = [0 for x in range(len(Emotions.EMOTIONS))]
            for sub_emotion, score in emotion_scores.items():
                x[Emotions.get_emotion_id(sub_emotion)] = score
            classifier_x.append(x)
            classifier_y.append(Emotions.get_emotion_id(emotion))

        self.engine.fit(classifier_x, classifier_y)
        self.trained = True

        if save_file:
            Storage.dump(self.filename, self.engine)

    def get_prediction(self, sentence):
        if not self.trained:
            engine = Storage.load(self.filename)
            if engine:
                self.engine = engine
                self.trained = True

        # Split into words
        words = self.text_utility.tokenize(sentence.lower())

        # Get the emotion scores for the words
        emotion_scores = self.word_emotion_scores.get_emotion_scores(words)

        x = [0 for x in range(len(Emotions.EMOTIONS))]
        for sub_emotion, score in emotion_scores.items():
            x[Emotions.get_emotion_id(sub_emotion)] = score

        return Emotions.get_emotion_for_id(self.engine.predict([x])[0])


class EmotionScoreSVM(EmotionScoreClassifier):
    """Trains an SVM based on the Emotion Scores of the Sentence"""

    FILENAME = 'svm_emotion_model'

    def __init__(self, allow_negation=False, filename=FILENAME):
        EmotionScoreClassifier.__init__(self, allow_negation, filename)
        self.engine = SVC()


class EmotionScoreGaussianNB(EmotionScoreClassifier):
    """Trains a gaussian naive bayes based on the Emotion Scores of the Sentence"""

    FILENAME = 'gaussian_nb_emotion_model'

    def __init__(self, allow_negation=False, filename=FILENAME):
        EmotionScoreClassifier.__init__(self, allow_negation, filename)
        self.engine = GaussianNB()


class EmotionScoreMultinomialNB(EmotionScoreClassifier):
    """Trains a multinomial naive bayes based on the Emotion Scores of the Sentence"""

    FILENAME = 'multinomial_nb_emotion_model'

    def __init__(self, allow_negation=False, filename=FILENAME):
        EmotionScoreClassifier.__init__(self, allow_negation, filename)
        self.engine = MultinomialNB()


class EmotionScoreBernoulliNB(EmotionScoreClassifier):
    """Trains a bernoulli naive classifier based on the Emotion Scores of the Sentence"""

    FILENAME = 'bernoulli_nb_emotion_model'

    def __init__(self, allow_negation=False, filename=FILENAME):
        EmotionScoreClassifier.__init__(self, allow_negation, filename)
        self.engine = BernoulliNB()


class EmotionScoreMaxEnt(EmotionScoreClassifier):
    """Trains a maximum entropy based on the Emotion Scores of the Sentence"""

    FILENAME = 'bernoulli_nb_emotion_model'

    def __init__(self, allow_negation=False, filename=FILENAME):
        EmotionScoreClassifier.__init__(self, allow_negation, filename)
        self.engine = LogisticRegression()


class MaxEmotionScore(SentenceLevel):
    """Returns the emotion with the maximum emotion score"""

    def __init__(self, allow_negation=False):
        SentenceLevel.__init__(self)
        self.allow_negation = allow_negation
        self.word_emotion_scores = WordEmotionScore(allow_negation=allow_negation)

    def get_prediction(self, sentence):
        # Split into words
        words = self.text_utility.tokenize(sentence.lower())

        # Get the emotion scores for the words
        emotion_scores = self.word_emotion_scores.get_emotion_scores(words)

        # get the best emotion - emotion with the highest score
        best_score = 0.0
        best_emotion = random.choice(Emotions.EMOTIONS)
        for emotion, score in emotion_scores.items():
            if score >= best_score:
                best_score = score
                best_emotion = emotion

        return best_emotion


class UnigramEmotionClassifier(SentenceLevel):
    """
    Classify using the feature set of emotion scores of a sentence
    """

    FOLDER = 'pickled/emotion/'
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
        words = self.text_utility.remove_stop_words(words)
        words = self.text_utility.lemmatize(words)
        return words

    def train_words(self, training_set):
        bar = ProgressBar()
        for tagged_line in bar(training_set):
            line = tagged_line[0]
            words = self.get_normalized_words(line)

            for word in words:
                if word not in self.word_set.keys():
                    self.word_set[word] = len(self.word_set)

        keys = list(self.word_set.keys())
        random.shuffle(keys)
        keys = keys[:min(self.WORD_LIMIT, len(keys))]

        self.word_set.clear()
        for word in keys:
            self.word_set[word] = len(self.word_set)

    def train(self, training_set, save_file=True):
        classifier_x = []
        classifier_y = []

        self.train_words(training_set)

        bar = ProgressBar()
        for tagged_line in bar(training_set):
            line = tagged_line[0]
            emotion = tagged_line[1]
            words = self.get_normalized_words(line)
            x = self.create_word_vector(words)

            classifier_x.append(x)
            classifier_y.append(Emotions.get_emotion_id(emotion))

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
        word_vector = self.create_word_vector(words)
        emotion_id = self.engine.predict([word_vector])[0]

        return Emotions.get_emotion_for_id(emotion_id)


class UnigramEmotionSVM(UnigramEmotionClassifier):
    """
    Trains a support vector machine naive bayes classifier over a feature set of unigrams
    """
    FILENAME = 'svm_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        UnigramEmotionClassifier.__init__(self, allow_negation, filename)
        self.engine = SVC()


class UnigramEmotionGaussianNB(UnigramEmotionClassifier):
    """
    Trains a gaussian naive bayes classifier over a feature set of unigrams
    """
    FILENAME = 'gaussian_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        UnigramEmotionClassifier.__init__(self, allow_negation, filename)
        self.engine = GaussianNB()


class UnigramEmotionBernoulliNB(UnigramEmotionClassifier):
    """
    Trains a bernoulli naive bayes classifier over a feature set of unigrams
    """
    FILENAME = 'bernoulli_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        UnigramEmotionClassifier.__init__(self, allow_negation, filename)
        self.engine = BernoulliNB()


class UnigramEmotionMultinomialNB(UnigramEmotionClassifier):
    """
    Trains a multinomial naive bayes classifier over a feature set of unigrams
    """
    FILENAME = 'multinomial_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        UnigramEmotionClassifier.__init__(self, allow_negation, filename)
        self.engine = MultinomialNB()

class UnigramEmotionMaxEnt(UnigramEmotionClassifier):
    """
    Trains a maximum entropy classifier over a feature set of unigrams
    """
    FILENAME = 'multinomial_nb_word_model'

    def __init__(self, allow_negation=True, filename=FILENAME):
        UnigramEmotionClassifier.__init__(self, allow_negation, filename)
        self.engine = LogisticRegression()
