__author__ = 'bijoy'


class Emotions:
    """
    Class containing list of emotions and their Ids
    """
    EMOTIONS = ["anticipation", "fear", "anger", "disgust", "joy", "sadness", "surprise", "trust"]

    @staticmethod
    def get_emotion_id(emotion):
        """
        Returns the id for an emotion
        :param emotion: the emotion
        :return: the id
        """
        return Emotions.EMOTIONS.index(emotion)

    @staticmethod
    def get_emotion_for_id(id):
        """
        Returns the emotion for the id
        :param id: the id
        :return: the emotion
        """
        return Emotions.EMOTIONS[id]
