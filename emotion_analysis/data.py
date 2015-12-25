from collections import defaultdict
from utilities.storage import Storage
from common.base_classes import DataSet
from utilities.text import Text
import random

__author__ = 'bijoy'


class WordEmotionScore(DataSet):
    """
    Data set for the word emotion score data from
    NRC Hashtag Emotion Lexicon
    Copyright (C) 2012 National Research Council Canada (NRC)
    Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)
    """
    FILENAME = "pickled/data/word_emotion_score.pickled"
    DATA_SET = "datasets/emotion_lexicon/word_emotion_score.txt"

    def read(self):
        mapping = defaultdict(dict)

        reading_started = False
        data_set_file = open(self.DATA_SET, "r")
        for line in data_set_file:
            if "######" in line:
                reading_started = True
                continue
            elif not reading_started:
                continue
            else:
                emotion, word, score = line.split()
                mapping[word][emotion] = float(score)

        data_set_file.close()
        Storage.dump(self.FILENAME, mapping)
        return mapping

    def get_emotion_scores(self, words):
        """
        For a list of words, returns the scores of each emotion
        :param words: the list of words
        :return: a dict containing the sum of scores of each emotion
        """
        emotion_scores = defaultdict(lambda: 0)
        lower_words = [word.lower() for word in words]

        sign = 1
        for word in lower_words:
            if word in Text.NEGATIVE_WORDS and self.allow_negation:
                sign *= -1
            elif Text.is_punctuation(word) and self.allow_negation:
                sign = 1

            if word in self.mapping.keys():
                for emotion, score in self.mapping[word].items():
                    emotion_scores[emotion] += sign * score

        return emotion_scores


class TweetDataSet(DataSet):
    """
    Data set for the emotion tagged tweets from
    NRC Hashtag Emotion Lexicon (twitter source)
    Copyright (C) 2012 National Research Council Canada (NRC)
    Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)
    """

    FILENAME = "pickled/data/tweet_data_set.pickled"
    DATA_SET = "datasets/emotion_lexicon/tweet_data_set.txt"

    def read(self):
        mapping = defaultdict(list)

        data_set_file = open(self.DATA_SET, "r")
        for line in data_set_file:
            line = line[19:]
            sentence, emotion = line.split("::")
            emotion = emotion.strip()
            mapping[emotion].append(sentence)
        data_set_file.close()

        if self.normalize_classes:
            min_lines = float("inf")
            for emotion, lines in mapping.items():
                min_lines = min(min_lines, len(lines))

            for emotion in mapping.keys():
                random.shuffle(mapping[emotion])
                mapping[emotion] = mapping[emotion][:min_lines]

        Storage.dump(self.FILENAME, mapping)
        return mapping
