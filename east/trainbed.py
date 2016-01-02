from east.emotion_analysis.sentence_level import EmotionScoreMaxEnt, UnigramEmotionMaxEnt
from east.emotion_analysis.data import TweetDataSet
from east.sentiment_analysis.data import MovieReviewDataSet
from east.sentiment_analysis.sentence_level import UnigramSentimentSVM, BigramSentimentMaxEnt, BigramSentimentSVM

__author__ = 'bijoy'

# #####################EMOTION ANALYSIS TRAINING########################
tweet_data_set = TweetDataSet(test_mode=False)
'''
max_emotion = MaxEmotionScore()
max_emotion.train(tweet_data_set.get_training_set())

gaussian_word = UnigramEmotionGaussianNB()
gaussian_word.train(tweet_data_set.get_training_set())

multinomial_word = UnigramEmotionMultinomialNB()
multinomial_word.train(tweet_data_set.get_training_set())

bernoulli_word = UnigramEmotionBernoulliNB()
bernoulli_word.train(tweet_data_set.get_training_set())

gaussian_emotion = EmotionScoreGaussianNB()
gaussian_emotion.train(tweet_data_set.get_training_set())

multinomial_emotion = EmotionScoreMultinomialNB()
multinomial_emotion.train(tweet_data_set.get_training_set())

bernoulli_emotion = EmotionScoreBernoulliNB()
bernoulli_emotion.train(tweet_data_set.get_training_set())

svm_emotion = EmotionScoreSVM()
svm_emotion.train(tweet_data_set.get_training_set())

svm_word = UnigramEmotionSVM()
svm_word.train(tweet_data_set.get_training_set())
'''

max_ent = EmotionScoreMaxEnt()
max_ent.train(tweet_data_set.get_training_set())

max_ent = UnigramEmotionMaxEnt()
max_ent.train(tweet_data_set.get_training_set())

########################################################################

# ####################SENTIMENT ANALYSIS TRAINING#######################

movie_data = MovieReviewDataSet(test_mode=False)
'''
bigram_bernoulli_nb = BigramSentimentBernoulliNB()
bigram_bernoulli_nb.train(movie_data.get_training_set())

bigram_gaussian_nb = BigramSentimentGaussianNB()
bigram_gaussian_nb.train(movie_data.get_training_set())

bigram_multinomial_nb = BigramSentimentMultinomialNB()
bigram_multinomial_nb.train(movie_data.get_training_set())

max_sentiment = MaxSentimentScore()
max_sentiment.train(movie_data.get_training_set())

max_opinion = OpinionLexiconSentimentCount()
max_opinion.train(movie_data.get_training_set())

gaussian_word = UnigramSentimentGaussianNB()
gaussian_word.train(movie_data.get_training_set())

bernoulli_word = UnigramSentimentBernoulliNB()
bernoulli_word.train(movie_data.get_training_set())

multinomial_word = UnigramSentimentMultinomialNB()
multinomial_word.train(movie_data.get_training_set())

multinomial_word = UnigramSentimentMaxEnt()
multinomial_word.train(movie_data.get_training_set())
'''

svm_word = BigramSentimentSVM()
svm_word.train(movie_data.get_training_set())

max_ent = BigramSentimentMaxEnt()
max_ent.train(movie_data.get_training_set())

svm_word = UnigramSentimentSVM()
svm_word.train(movie_data.get_training_set())

#######################################################################
