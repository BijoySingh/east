from collections import defaultdict
import re

from east.utilities.storage import Storage
from east.common.base_classes import DataSet

__author__ = 'bijoy'


class MovieReviewDataSet(DataSet):
    """
    The data set for the movie reviews, for the
    v1.0 sentence polarity data set comes from the URL
    http://www.cs.cornell.edu/people/pabo/movie-review-data .
    """

    FILENAME = DataSet.FOLDER + "movie_review_data_set.pickled"
    POSITIVE_DATA_SET = "datasets/movie_reviews/positive_reviews.txt"
    NEGATIVE_DATA_SET = "datasets/movie_reviews/negative_reviews.txt"

    def read(self):
        mapping = defaultdict(list)

        data_set_file = open(self.get_absolute_path(self.POSITIVE_DATA_SET), "r")
        for line in data_set_file:
            mapping["positive"].append(line)
        data_set_file.close()

        data_set_file = open(self.get_absolute_path(self.NEGATIVE_DATA_SET), "r")
        for line in data_set_file:
            mapping["negative"].append(line)
        data_set_file.close()

        Storage.dump(self.get_absolute_path(self.FILENAME), mapping)
        return mapping


class OpinionLexicon(DataSet):
    """
    The data from the opinion lexicon,
    Available as 'A list of positive and negative opinion words or
    sentiment words for English' from
    http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets
    """

    FILENAME = DataSet.FOLDER + "opinion_lexicon_data_set.pickled"
    POSITIVE_DATA_SET = "datasets/opinion_lexicon/positive-words.txt"
    NEGATIVE_DATA_SET = "datasets/opinion_lexicon/negative-words.txt"

    def read(self):
        mapping = dict()
        started_reading = False
        data_set_file = open(self.get_absolute_path(self.POSITIVE_DATA_SET), "r")
        for line in data_set_file:
            if line.strip() == '####START####':
                started_reading = True
            elif started_reading:
                mapping[line.strip()] = 'positive'
        data_set_file.close()

        started_reading = False
        data_set_file = open(self.get_absolute_path(self.NEGATIVE_DATA_SET), "r")
        for line in data_set_file:
            if line.strip() == '####START####':
                started_reading = True
            elif started_reading:
                mapping[line.strip()] = 'negative'
        data_set_file.close()

        Storage.dump(self.FILENAME, mapping)
        return mapping

    def get_opinion_count(self, words):
        """
        Returns the count of each sentiment words based on the opinion lexicon
        :param words: the list of words
        :return: a list [a, b] with a : number of negative words, b : number of positive words
        """
        sentiment_counts = [0, 0]
        if self.allow_negation:
            negated_words = self.get_negated_words(words)
        else:
            negated_words = [(word.lower(), False) for word in words]
        for word, negated in negated_words:
            if word in self.mapping.keys():
                sentiment = self.mapping[word]
                if sentiment == 'positive' or (sentiment == 'negative' and negated):
                    sentiment_counts[1] += 1
                elif sentiment == 'negative' or (sentiment == 'positive' and negated):
                    sentiment_counts[0] += 1
        return sentiment_counts


class SentiWordNet(DataSet):
    FILENAME = DataSet.FOLDER + "senti_word_net_data_set.pickled"
    DATA_SET = "datasets/sentiwordnet/SentiWordNet_3.0.0.txt"

    def read(self):
        mapping = dict()

        data_set_file = open(self.get_absolute_path(self.DATA_SET), "r")
        for line in data_set_file:
            if line[0] == '#':
                continue

            columns = line.split('\t')
            pos_score = float(columns[2])
            neg_score = float(columns[3])
            words = columns[4].split()

            for word in words:
                word = re.sub(r'#[0-9]', "", word)
                mapping[word] = (neg_score, pos_score)

        data_set_file.close()

        Storage.dump(self.get_absolute_path(self.FILENAME), mapping)
        return mapping

    def get_sentiment_scores(self, words):
        """
        Returns the sentiment scores for each of the sentiment based on the SentiWordNet
        :param words: the list of words
        :return: a list [a, b] with a : number of negative sentiment score,
         b : number of positive sentiment score
        """
        sentiment_scores = [0, 0]
        if self.allow_negation:
            negated_words = self.get_negated_words(words)
        else:
            negated_words = [(word.lower(), False) for word in words]

        for word, negated in negated_words:
            sign = -1 if negated else 1
            if word in self.mapping.keys():
                sentiments = self.mapping[word]
                sentiment_scores[0] += sign * sentiments[0]
                sentiment_scores[1] += sign * sentiments[1]

        return sentiment_scores
