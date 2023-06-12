# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 23:47:34 2023

@author: janba
"""
import tweetnlp
import pandas as pd

file_loc = 'D:\\Code\\Text_mining\\NewsAPI\\Data\\'

data = pd.read_csv(file_loc + 'NRC_tweets_van_23_05_03_tot_23_06_08.csv')

# test_tweet = data.processed_tweet[0]
# print(test_tweet)

model = tweetnlp.load_model('sentiment', multilingual=True)

# sent = model.sentiment(test_tweet, return_probability=True)

# print(sent)
data['Sentiment'] = ''
for i, tweet in enumerate(data.processed_tweet):
    if i % 100 == 0: #Every 100 times
        data.to_csv(file_loc + 'tweets_w_sentiment.csv')
    data.at[i, 'Sentiment'] = model.sentiment(tweet, return_probability=True)
    
data.to_csv(file_loc + 'tweets_w_sentiment.csv')