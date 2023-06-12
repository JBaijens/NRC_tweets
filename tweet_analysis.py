# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:33:34 2023

@author: Jan Baijens

This script makes data visualizations of twitter data using matplotlib-pyplot and seaborn.

Plots to make:
Timeline number of tweets V
Timeline sentiment analysis V
Associated words wordweb V
Most discussed articles, nrc tweets
Sentiment analysis comparison with other newspaper
Retweets, replies, likes per tweet sentiment (also comparison other newspaper?)
NRC tweet count per time of day V
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
import re
import itertools
from collections import Counter
import nltk
import numpy as np

#Setup paths:
current_dir = os.getcwd()
Main_dir = Path(os.path.split(current_dir)[0])
Data_dir = Main_dir / 'Data'
Figures_dir = Main_dir / 'Figures'

#Define functions:
def formatter(x, pos):
    return str(round(x / 1e6, 1)) + " M"

def preprocess_tweet(input_tweet):
    tweet = str(input_tweet).lower() #Make everything lowercase
    # tweet = re.sub(r'[^\w\s]', '', tweet) 
    tweet = re.sub(r"[^a-zA-Z0-9 ]", "", tweet) #Remove special characters
    words = []
    for word in tweet.split(' '):
        if word.startswith('http'):
            word = ''
        if len(word) > 0:
            words.append(word)
    return ' '.join(words), words #Return preprocessed words as string, and a list of words

#Load data

#Load data:
data = pd.read_csv(str(Data_dir / 'tweets_w_sentiment.csv'), index_col=0)
data.Date = pd.to_datetime(data.Date, utc=True) #From string to datetime object, is useful for grouping by date.
data.words = [eval(i) for i in data.words] #From string to list

#Add columns to dataset:
data['Sentimentlabel'] = [eval(data.Sentiment[i])['label'] for i in range(len(data))]
data['Sentimentprob_neg'] = [eval(data.Sentiment[i])['probability']['negative'] for i in range(len(data))]
data['Sentimentprob_neu'] = [eval(data.Sentiment[i])['probability']['neutral'] for i in range(len(data))]
data['Sentimentprob_pos'] = [eval(data.Sentiment[i])['probability']['positive'] for i in range(len(data))]

#Check for redundancy in dataset:
len(set(data.Tweetid)) #None found, tweets are unique

#Preprocess tweets:
processed_tweets, tweet_words = [], []
for tweet in data.Tweet:
    processed_tweet, words = preprocess_tweet(tweet)
    processed_tweets.append(processed_tweet)
    tweet_words.append(words)
    
data['processed_tweet'], data['words'] = processed_tweets, tweet_words

#Count tweets, views:
total_views_per_day = data.groupby(data.Date.dt.date)['Viewcount'].sum()
total_tweets_per_day = data.groupby(data.Date.dt.date)['Tweet'].count()

total_views_per_hour = data.groupby(data.Date.dt.hour)['Viewcount'].sum()
total_tweets_per_hour = data.groupby(data.Date.dt.hour)['Tweet'].count()

data_by_date = data.groupby(data.Date.dt.date)
date_label_counts = []
for day in set(data.Date.dt.date):
    day_data = data_by_date.get_group(day)
    day_total_count = day_data.Tweetid.count()
    for label in ['negative', 'neutral', 'positive']:
        day_label_count = day_data.loc[day_data.Sentimentlabel == label]['Tweetid'].count()
        date_label_counts.append({'day': day, 
                                  'label' : label, 
                                  'Tweetcount' : day_label_count,
                                  'percentage' : day_label_count / day_total_count})
        
date_label_counts_df = pd.DataFrame(date_label_counts)
date_label_counts_df_pos = date_label_counts_df.loc[date_label_counts_df.label == 'positive']
positive_tweets = data[data.Sentimentlabel == 'positive']

#Plot number of tweets per day, color by sentiment:
hist_fig_size = (10, 5)
sns.set(rc={'figure.figsize': hist_fig_size})
sns.set_theme(style="ticks")
ax = sns.histplot(date_label_counts_df, x = 'day', hue = 'label', weights = 'Tweetcount', 
                  multiple = 'stack', palette = 'tab20c', legend = True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d-%m"))
plt.xlabel('')
plt.ylabel('Aantal tweets')
ax.legend(labels = ['Negatief', 'Neutraal', 'Positief'], title = 'Sentiment') #Color order in legend is wrong
sns.despine()
plt.savefig(str(Figures_dir / 'Tweetcount_date.png'), dpi =300)
plt.show()

#Plot number of views per day:
ax = sns.histplot(x = total_views_per_day.index, weights = total_views_per_day.values, 
                  multiple = 'stack', color = sns.color_palette('tab20c')[4], legend = False)
ax.invert_yaxis()
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d-%m"))
plt.xlabel('')
plt.ylabel('Aantal views')
# ax.legend(labels = ['Negatief', 'Neutraal', 'Positief'], title = 'Sentiment')
# sns.despine()
plt.savefig(str(Figures_dir / 'Viewcount_date.png'), dpi =300)
plt.show()

#Plot overview of views:
views_data = []
cutoff = 112000
for index, row in data.iterrows():
    viewcount = row.Viewcount
    user = 'Overige'
    if viewcount > cutoff:
        user = row.Userdisplayname
        print(viewcount, user)
    views_data.append({'user' : user, 'x' : 1, 'views' : viewcount, 'text' : row.Tweet})
views_data = pd.DataFrame(views_data)   
views_data.sort_values('views', inplace = True, ascending = True) 
views_data_subset = views_data.loc[views_data.views >= cutoff].copy()

#Get total views per user:
viewcount_per_user = data.groupby('Userdisplayname')['Viewcount'].sum()
viewcount_per_user_subset = viewcount_per_user.loc[lambda x: x >= cutoff].copy()
viewcount_per_user_subset.sort_values(inplace = True, ascending = False)
viewcount_per_user_subset = viewcount_per_user_subset.append(pd.Series([viewcount_per_user.sum() - viewcount_per_user_subset.sum()], index = ['Overige']))
df = viewcount_per_user_subset.to_frame()
df['y'] = 1

df_plot_order = list(df.index)
ax = sns.histplot(data = df, y = 'y', hue = df.index, 
                   # hue_order = df_plot_order[::-1], 
                  weights = 0, 
                  bins = 3, multiple = 'stack', palette = 'tab20c', legend = True)
# handles, labels = ax.get_legend_handles_labels()
legend = ax.get_legend()
legend.set_bbox_to_anchor((1, 1))
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=90)
# plt.axis('off')
plt.title('Weergaven per gebruiker')
plt.savefig(str(Figures_dir / 'user_total_views.png'), dpi = 300, bbox_inches="tight")
plt.show()

#Split dataset in NRC tweets and tweets from other users:
NRC_tweets = data.loc[data['Username'] == 'nrc'].copy()
other_tweets = data.loc[data['Username'] != 'nrc'].copy()
other_user_counts = dict(Counter(other_tweets.Username))

NRC_tweets_words = list(itertools.chain.from_iterable(NRC_tweets.words))
banned_words = nltk.corpus.stopwords.words('dutch') #Woorden zoals die, de, het, en, etc.
NRC_tweets_filtered_words = [word for word in NRC_tweets_words if word not in banned_words]
NRC_tweets_word_counts = dict(Counter(NRC_tweets_filtered_words))

#Plot number of tweets per hour of day:
ax = sns.barplot(x = total_tweets_per_hour.index, 
                 y = total_tweets_per_hour.values, 
                 color = sns.color_palette('tab20c')[8])    
plt.xlabel('Uur van de dag')
plt.ylabel('Aantal tweets')
plt.savefig(str(Figures_dir / 'Tweetcount_hour_of_day.png'), dpi =300)
plt.show()

#Count words
all_words = list(itertools.chain.from_iterable(data.words))
filtered_words = [word for word in all_words if word not in banned_words]
filtered_text = ' '.join(filtered_words)
word_counts = pd.Series(Counter(filtered_words))

#Analyse tweet sentiments:
sentiment_counts = dict(Counter(data.Sentimentlabel))
data_by_replyid = data.groupby(data.Inreplytweetid)
