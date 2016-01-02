from east.emotion_analysis.sentence_level import MaxEmotionScore, EmotionScoreSVM, UnigramEmotionSVM, \
    UnigramEmotionGaussianNB, UnigramEmotionBernoulliNB, UnigramEmotionMultinomialNB, EmotionScoreGaussianNB, \
    EmotionScoreBernoulliNB, EmotionScoreMultinomialNB
from east.emotion_analysis.data import TweetDataSet
from east.sentiment_analysis.data import MovieReviewDataSet
from east.sentiment_analysis.sentence_level import UnigramSentimentSVM, MaxSentimentScore, \
    UnigramSentimentBernoulliNB, UnigramSentimentGaussianNB, UnigramSentimentMultinomialNB,\
    OpinionLexiconSentimentCount, UnigramSentimentMaxEnt, BigramSentimentBernoulliNB,\
    BigramSentimentMultinomialNB, BigramSentimentGaussianNB

__author__ = 'bijoy'

# #####################EMOTION ANALYSIS TESTBED########################

tweet_data_set = TweetDataSet(test_mode=True)

# MaxEmotionScore : 37.44%
max_emotion = MaxEmotionScore()
max_emotion.test(tweet_data_set.get_testing_set())

# WordEmotionGaussianNB : 40.04%
gaussian_word = UnigramEmotionGaussianNB()
gaussian_word.test(tweet_data_set.get_testing_set())

# WordEmotionMultinomialNB : 44.80%
multinomial_word = UnigramEmotionMultinomialNB()
multinomial_word.test(tweet_data_set.get_testing_set())

# WordEmotionBernoulliNB : 44.15%
bernoulli_word = UnigramEmotionBernoulliNB()
bernoulli_word.test(tweet_data_set.get_testing_set())

# EmotionScoreGaussianNB : 44.15%
gaussian_emotion = EmotionScoreGaussianNB()
gaussian_emotion.test(tweet_data_set.get_testing_set())

# EmotionScoreMultinomialNB : 46.53%
multinomial_emotion = EmotionScoreMultinomialNB()
multinomial_emotion.test(tweet_data_set.get_testing_set())

# EmotionScoreBernoulliNB : 24.02%
bernoulli_emotion = EmotionScoreBernoulliNB()
bernoulli_emotion.test(tweet_data_set.get_testing_set())

# EmotionScoreSVM : 48.26%
svm_emotion = EmotionScoreSVM()
svm_emotion.test(tweet_data_set.get_testing_set())

# WordEmotionSVM : 31.16%
svm_word = UnigramEmotionSVM()
svm_word.test(tweet_data_set.get_testing_set())

#######################################################################

# ####################SENTIMENT ANALYSIS TESTBED#######################
movie_data = MovieReviewDataSet()

# BigramSentimentBernoulliNB : 61.52%
bigram_bernoulli_nb = BigramSentimentBernoulliNB()
bigram_bernoulli_nb.test(movie_data.get_testing_set())

# BigramSentimentGaussianNB : 58.14%
bigram_gaussian_nb = BigramSentimentGaussianNB()
bigram_gaussian_nb.test(movie_data.get_testing_set())

# BigramSentimentMultinomialNB : 61.23%
bigram_multinomial_nb = BigramSentimentMultinomialNB()
bigram_multinomial_nb.test(movie_data.get_testing_set())

# MaxSentimentScore : 44.38%
max_sentiment = MaxSentimentScore()
max_sentiment.test(movie_data.get_testing_set())

# OpinionLexiconSentimentCount : 63.29%
max_opinion = OpinionLexiconSentimentCount()
max_opinion.test(movie_data.get_testing_set())

# WordSentimentSVM : 59.26%
svm_word = UnigramSentimentSVM()
svm_word.test(movie_data.get_testing_set())

# WordSentimentGaussianNB : 65.26%
gaussian_word = UnigramSentimentGaussianNB()
gaussian_word.test(movie_data.get_testing_set())

# WordSentimentBernoulliNB : 77.52%
bernoulli_word = UnigramSentimentBernoulliNB()
bernoulli_word.test(movie_data.get_testing_set())

# WordSentimentMultinomialNB : 77.90%
multinomial_word = UnigramSentimentMultinomialNB()
multinomial_word.test(movie_data.get_testing_set())

# WordSentimentMaxEnt : 75.56%
multinomial_word = UnigramSentimentMaxEnt()
multinomial_word.test(movie_data.get_testing_set())

#######################################################################