import pickle
import gzip

__author__ = 'bijoy'


class Storage:
    DOT_ZIP = '.zip'

    @staticmethod
    def dump(path, variable):
        """
        Dumps a variable to the path
        :param path: the file to be stored at
        :param variable: the variable to be stored
        :return: None
        """
        with gzip.open(path + Storage.DOT_ZIP, 'wb') as f:
            pickle.dump(variable, f)

    @staticmethod
    def load(path):
        """
        Tries to load a value from a pickled file
        :param path: the path of the pickle file
        :return: the variable
        """
        try:
            with gzip.open(path + Storage.DOT_ZIP, 'rb') as f:
                return pickle.load(f)
        except:
            return None
