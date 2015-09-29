from __future__ import print_function

import string
import json
import sys
import itertools

from pyspark import SparkContext,SparkFiles

# these are ripped out of NLTK so we don't have to download any packages on worker nodes
PUNCTUATION = set(['!', '#', '"', '%', '$', "'", '&', ')', '(', '+', '*', '-', ',', '/', '.', ';', ':', '=', '<', '?', '>', '@', '[', ']', '\\', '_', '^', '`', '{', '}', '|', '~'])
STOPWORDS = set([u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'herself', u'had', u'should', u'to', u'only', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'did', u'this', u'she', u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'while', u'does', u'above', u'between', u't', u'be', u'we', u'who', u'were', u'here', u'hers', u'by', u'on', u'about', u'of', u'against', u's', u'or', u'own', u'into', u'yourself', u'down', u'your', u'from', u'her', u'their', u'there', u'been', u'whom', u'too', u'themselves', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'these', u'up', u'will', u'below', u'can', u'theirs', u'my', u'and', u'then', u'is', u'am', u'it', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'yours', u'so', u'the', u'having', u'once'])

# Generates tokens for a text blob, then does some massaging
# on each token (lower casing, removing punctuation, filtering
# stopwords) before returning the list of tokens.
def tokenize(text):
    tokens = text.split()
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    return no_stopwords

# Takes a tweet, then returns the tweet and a list of
# of the tokens in it
def tokenize_tweet(tweet):
	return (tweet, tokenize(tweet))

# Scores a tweet based on the number of tokens that appear
# in the supplied word list. The returned score is the
# number of matching tokens divided by the length of the
# token list. This bounds the return value from 0.0 to 1.0.
def score_tokens(tokens, word_list):
	if len(tokens) == 0:
		return 0
	matches = [t for t in tokens if t in word_list]
	score = float(len(matches)) / float(len(tokens))
	return score

# Scores a tweet/token pair using the provided
# positive and negative word lists. 
def score_tweet(pair, positive_words, negative_words):
	tweet = pair[0]
	tokens = pair[1]
	pos_score = score_tokens(tokens, positive_words)
	neg_score = score_tokens(tokens, negative_words)
	if pos_score == neg_score:
		return (tweet, 0)
	elif pos_score > neg_score:
		return (tweet, pos_score)
	else:
		return (tweet, -1.0 * neg_score)

# This function sets up the positive/negative words set then
# lazily maps the partition to the score function
def score_tweets(iterator):
	positive_words = [line.strip() for line in open(SparkFiles.get("pos-words.txt"))]
	negative_words = [line.strip() for line in open(SparkFiles.get("neg-words.txt"))]
	return itertools.imap(lambda x: score_tweet(x, positive_words, negative_words), iterator)	

# Main driver.
sc = SparkContext(appName="Sentiment Analysis Demo") 
rrd = sc.textFile("hdfs:///tweets")
tweets = rrd.map(lambda line: json.loads(line)).filter(lambda tweet: 'text' in tweet).map(lambda tweet: tweet['text'])
scored_tweets = tweets.map(tokenize_tweet).mapPartitions(score_tweets)
scored_tweets.saveAsTextFile("hdfs:///scored_tweets")
