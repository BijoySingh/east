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

# MaxEmotionScore :
max_emotion = MaxEmotionScore()
tweet_data_set.test_algorithm(max_emotion)

# WordEmotionGaussianNB :
gaussian_word = UnigramEmotionGaussianNB()
tweet_data_set.test_algorithm(gaussian_word)

# WordEmotionMultinomialNB :
multinomial_word = UnigramEmotionMultinomialNB()
tweet_data_set.test_algorithm(multinomial_word)

# WordEmotionBernoulliNB :
bernoulli_word = UnigramEmotionBernoulliNB()
tweet_data_set.test_algorithm(bernoulli_word)

# EmotionScoreGaussianNB :
gaussian_emotion = EmotionScoreGaussianNB()
tweet_data_set.test_algorithm(gaussian_emotion)

# EmotionScoreMultinomialNB :
multinomial_emotion = EmotionScoreMultinomialNB()
tweet_data_set.test_algorithm(multinomial_emotion)

# EmotionScoreBernoulliNB :
bernoulli_emotion = EmotionScoreBernoulliNB()
tweet_data_set.test_algorithm(bernoulli_emotion)

# EmotionScoreSVM :
svm_emotion = EmotionScoreSVM()
tweet_data_set.test_algorithm(svm_emotion)

# WordEmotionSVM :
svm_word = UnigramEmotionSVM()
tweet_data_set.test_algorithm(svm_word)

#######################################################################

# ####################SENTIMENT ANALYSIS TESTBED#######################
movie_data = MovieReviewDataSet()

# BigramSentimentBernoulliNB :
bigram_bernoulli_nb = BigramSentimentBernoulliNB()
movie_data.test_algorithm(bigram_bernoulli_nb)

# BigramSentimentGaussianNB :
bigram_gaussian_nb = BigramSentimentGaussianNB()
movie_data.test_algorithm(bigram_gaussian_nb)

# BigramSentimentMultinomialNB :
bigram_multinomial_nb = BigramSentimentMultinomialNB()
movie_data.test_algorithm(bigram_multinomial_nb)

# MaxSentimentScore :
max_sentiment = MaxSentimentScore()
movie_data.test_algorithm(max_sentiment)

# OpinionLexiconSentimentCount :
max_opinion = OpinionLexiconSentimentCount()
movie_data.test_algorithm(max_opinion)

# WordSentimentSVM :
svm_word = UnigramSentimentSVM()
movie_data.test_algorithm(svm_word)

# WordSentimentGaussianNB :
gaussian_word = UnigramSentimentGaussianNB()
movie_data.test_algorithm(gaussian_word)

# WordSentimentBernoulliNB :
bernoulli_word = UnigramSentimentBernoulliNB()
movie_data.test_algorithm(bernoulli_word)

# WordSentimentMultinomialNB :
multinomial_word = UnigramSentimentMultinomialNB()
movie_data.test_algorithm(multinomial_word)

# WordSentimentMaxEnt :
max_ent = UnigramSentimentMaxEnt()
movie_data.test_algorithm(max_ent)

#######################################################################