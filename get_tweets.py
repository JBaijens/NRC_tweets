# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:30:52 2023

@author: janba

This script obtains tweets referencing dutch newspapers from Twitter using snscrape, 
and stores these tweets as csv file.
"""
import snscrape.modules.twitter as sntwitter
import pandas as pd


query = 'NRC lang:nl until:2023-06-09 since:2023-06-05' 
tweets = []
limit = 5000

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()): #Twitter API unstable, might need to run multiple times
    if i >= limit:
        break
    if i%1000 == 0: 
        print(i) #Print an update every 1000 tweets
    tweets.append([tweet.date, tweet.id,
                   tweet.user.username, tweet.user.id, tweet.user.displayname, tweet.user.followersCount,
                   tweet.rawContent, tweet.inReplyToTweetId,
                   tweet.hashtags, tweet.replyCount, tweet.retweetCount, tweet.likeCount, tweet.viewCount])
        
df = pd.DataFrame(tweets, columns=['Date', 'Tweetid', 'Username', 'Userid', 'Userdisplayname', 'Userfollowercount', 
                                   'Tweet', 'Inreplytweetid', 'Hashtags', 'Replycount', 'Retweetcount', 'Likecount', 
                                   'Viewcount'])

df.to_csv('4487_tweets_NRC_until_23_06_08_since_23_06_05.csv')
