# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:07:02 2023

@author: janba
"""
import pandas as pd
import itertools
from collections import Counter
import nltk
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

file_loc = 'D:\\Code\\Text_mining\\NewsAPI\\Data\\'
data = pd.read_csv(file_loc + 'NRC_tweets_van_23_05_03_tot_23_06_08.csv')
print('Finished reading data, found {} rows'.format(len(data)))

#Count words
all_word_lists = [eval(i) for i in data.words]
all_words = list(itertools.chain.from_iterable(all_word_lists))
banned_words = nltk.corpus.stopwords.words('dutch') #Woorden zoals die, de, het, en, etc.
banned_words += ['we', 'via', 'gaat', 'waar', 'wel', 'weer', 'n'] #Add extra words with low information
filtered_words = [word for word in all_words if word not in banned_words]
filtered_text = ' '.join(filtered_words)
word_counts = pd.Series(Counter(filtered_words))
print('Finished counting words, found {} words after filtering'.format(len(filtered_words)))

def generate_wordcloud(data, title, mask=None):
    cloud = WordCloud(scale=3,
                      max_words=150,
                      # colormap='RdYlGn',
                      color_func = colors,
                      mask=mask,
                      background_color='white',
                      # stopwords=banned_words,
                      # contour_color='black',
                      # contour_width=1,
                      collocations=False).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.savefig(file_loc + 'wordcloud_NRC.png', dpi = 300)
    plt.show()
    
nrc_logo_file = str(file_loc + 'NRC_logo_white_background.png')
nrc_logo = np.array(Image.open(nrc_logo_file))
colors = ImageColorGenerator(nrc_logo)
generate_wordcloud(filtered_text, '', nrc_logo)