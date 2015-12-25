__author__ = 'bijoy'
from utilities.text import Text
from utilities.storage import Storage
from collections import defaultdict
from progressbar import ProgressBar


class DataSet:
    """
    The base class for any data set
    """

    FOLDER = 'pickled/data/'
    FILENAME = 'data_set.pickled'

    def __init__(self, k=5, test_mode=True, allow_negation=True, normalize_classes=True):
        """
        The data set default constructor
        :param k: the k-fold cross validation's k
        :param test_mode: test mode true means that k will be considered,
         else k will not be considered.
        :param allow_negation: allows negation in the scoring (if applicable)
        :param normalise_classes: clips the data points in classes which have more data points
         to equalise the number of data points for all the classes
        :return: None
        """

        # mapping variable is the key variable in the class.
        # It must be a 'dict' object with each emotion/sentiment class mapping to a list of sentences.
        self.mapping = None

        self.k = k
        self.test_mode = test_mode
        self.allow_negation = allow_negation
        self.normalize_classes = normalize_classes
        self.load()

    def load(self):
        """
        Loads the data set from the variable / data file / pickled file
        :return: the data set
        """
        if not self.mapping:
            self.mapping = Storage.load(self.FILENAME)
            if not self.mapping:
                self.mapping = self.read()

    def read(self):
        """
        Reads the data from the data file
        (For new classes, this needs to be implemented, due to the fact that data sets are not freely parsable)
        :return: the data set
        """
        pass

    def test_algorithm(self, algorithm, print_result=True):
        """
        Tests an algorithm using k-fold cross validation
        :param algorithm: the algorithm object
        :param print_result: print the result?
        :return:
        """
        result = dict()
        result["count"] = 0
        result["accuracy"] = 0
        result["precision"] = defaultdict(lambda: 0)
        result["recall"] = defaultdict(lambda: 0)

        print(algorithm.__class__.__name__)
        for k in range(self.k):
            print("Step " + str(result["count"] + 1) + " of " + str(self.k))
            algorithm.train(self.get_training_set(k), False)
            k_result = algorithm.test(self.get_testing_set(k), False)
            result["count"] += 1
            result["accuracy"] += k_result["accuracy"]
            for key, item in k_result["precision"].items():
                result["precision"][key] += item
            for key, item in k_result["recall"].items():
                result["recall"][key] += item

        count = float(result["count"])

        result["accuracy"] = float(result["accuracy"]) / count
        for key, item in result["precision"].items():
            result["precision"][key] = float(item) / count
        for key, item in result["recall"].items():
            result["recall"][key] = float(item) / count

        if print_result:
            SentenceLevel.print_result(result)

        return result

    def get_training_set(self, training_set_id=0):
        """
        Gets the training set from the data set - first (k-1 / k)th of the data
        :return: the subset of the data set usable for training.
        """
        training_set = list()
        if not self.test_mode:
            for class_id, lines in self.mapping.items():
                training_set += [(line, class_id) for line in lines]
            return training_set

        ratio = float(1) / float(self.k)
        first_set_end = training_set_id * ratio
        last_set_start = (training_set_id + 1) * ratio

        training_set = list()
        for class_id, lines in self.mapping.items():
            first_end = int(first_set_end * len(lines))
            last_start = int(last_set_start * len(lines))
            training_set += [(line, class_id) for line in lines[:first_end]]
            training_set += [(line, class_id) for line in lines[last_start:]]

        return training_set

    def get_testing_set(self, testing_set_id=0):
        """
        Gets the testing set from the data set - last (1 / k)th of the data
        :return: the subset of the data set usable for testing.
        """
        if not self.test_mode:
            return self.get_testing_set(testing_set_id)

        ratio = float(1) / float(self.k)
        test_set_start = testing_set_id * ratio
        test_set_end = (testing_set_id + 1) * ratio

        testing_set = list()
        for emotion, lines in self.mapping.items():
            set_start = int(test_set_start * len(lines))
            set_end = int(test_set_end * len(lines))
            testing_set += [(line, emotion) for line in lines[set_start:set_end]]

        return testing_set

    @staticmethod
    def get_negated_words(words):
        """
        Given a list of words, returns a list of tuples indicating whether it should be negated
        :param words: the list of words
        :return: the list of tuples
        """
        lower_words = [word.lower() for word in words]
        negated_words = []
        negate = False
        for word in lower_words:
            if word in Text.NEGATIVE_WORDS:
                negate = not negate
            elif Text.is_punctuation(word):
                negate = False

            negated_words.append((word, negate))
        return negated_words


class SentenceLevel:
    """
    Base class for the sentence level sentiment and emotion analysis algorithms.
    """
    FOLDER = 'pickled/'
    FILENAME = 'default'

    def __init__(self, filename=FILENAME):
        self.filename = self.FOLDER + filename + ".pickled"
        self.text_utility = Text()

    def train(self, training_set, save_file=True):
        """
        Train the model (if the model needs training)
        (For new classes, this needs to be implemented)
        :param training_set: the training set
        :param save_file: save the training set in the file
        :return: None
        """
        pass

    def get_prediction(self, sentence):
        """
        Get the prediction for the sentence
        (For new classes, this needs to be implemented)
        :param sentence:  the sentence
        :return: the prediction
        """
        pass

    def test(self, testing_set, should_print=True):
        """
        Test the model over a testing set
        :param testing_set: the testing set to test the algorithm over
        :param should_print: whether the result should be printed or not
        :return: the result dict object
        """
        results = list()
        precision = defaultdict(lambda: defaultdict(lambda: 0))
        recall = defaultdict(lambda: defaultdict(lambda: 0))
        match_count = 0

        bar = ProgressBar()
        for tagged_line in bar(testing_set):
            line = tagged_line[0]
            expected_emotion = tagged_line[1]
            predicted_emotion = self.get_prediction(line)
            results.append((expected_emotion, predicted_emotion))
            is_same = expected_emotion == predicted_emotion
            precision[predicted_emotion][is_same] += 1
            recall[expected_emotion][is_same] += 1
            match_count = match_count + 1 if is_same else match_count

        result = {"accuracy": 100.0 * match_count / len(testing_set),
                  "precision": precision,
                  "recall": recall}

        for key in result["precision"].keys():
            true_count = result["precision"][key][True]
            false_count = result["precision"][key][False]
            result["precision"][key] = true_count * 100.0 / (true_count + false_count)

        for key in result["recall"].keys():
            true_count = result["recall"][key][True]
            false_count = result["recall"][key][False]
            result["recall"][key] = true_count * 100.0 / (true_count + false_count)

        if should_print:
            self.print_result(result)

        return result

    @staticmethod
    def print_result(result):
        """
        Prints the results of the testing
        :param result: The Result Object
        :return:
        """
        print("accuracy : " + str(result["accuracy"]))
        print("precision : ")
        for key, item in result["precision"].items():
            print("    " + key + " : " + str(item))
        print("recall : ")
        for key, item in result["recall"].items():
            print("    " + key + " : " + str(item))


class DocumentLevel:
    """
    Base class for document level analysis
    """

    def get_prediction(self, tags=None, sentences=None):
        """
        Gets the class for a document
        :param tags: the list of tags for the sentences
        :param sentences: the sentences
        :return the tag
        """
        pass

    @staticmethod
    def get_most_continuous_tag(tags):
        """
        Returns the tags occurring in the longest continuous sequence
        :param tags: the list of tags
        :return the tag
        """
        most_continuous_tag = None
        most_continuous_tag_count = 0

        last_tag = None
        last_tag_count = 0

        for tag in tags:
            if tag == last_tag:
                last_tag_count += 1
            else:
                last_tag = tag
                last_tag_count = 1

            if most_continuous_tag_count <= last_tag_count:
                most_continuous_tag_count = last_tag_count
                most_continuous_tag = last_tag

        return most_continuous_tag

    @staticmethod
    def get_max_tag(tags):
        """
        Returns the tag most present in the list of tags
        :param tags: the list of tags
        :return: the most occuring tag
        """
        tag_count = defaultdict(lambda: 0)
        for tag in tags:
            tag_count[tag] += 1

        max_tag_count = max(tag_count.values())

        max_tag = None
        for tag, count in tag_count.items():
            if count == max_tag_count:
                max_tag = tag

        return max_tag