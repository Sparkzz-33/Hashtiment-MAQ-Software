#! C:/Users/rajpu/AppData/Local/Programs/Python/Python38/python.exe

import sys
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from twitterscraper import query_tweets
from datetime import datetime
import datetime as dt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
#import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import twint

def program_sentiment(hashtag):
	return json.dumps(hashtag + " hello you successfully executed python")

if __name__ == "__main__":
	print("Hello Work")
	def detector(x):
		try:
			return detect(x)
		except:
			None
	analyzer = SentimentIntensityAnalyzer()
	f = open("demo.txt", "w")
	start = str(sys.argv[3])
	f.write(sys.argv[1] + "\n")
	f.write(sys.argv[2] + "\n")
	f.write(start + "\n")
	start = start.replace('-', "")
	end = str(sys.argv[4])
	f.write(end)
	f.close()
	end = end.replace('-', "")
	format_str = "%Y%m%d"
	#print(start)
	begin_date = dt.datetime.strptime(start, format_str)
	begin_date = begin_date.strftime(format_str)
	#print(begin_date)
	end_date = dt.datetime.strptime(end, format_str)
	end_date = end_date.strftime(format_str)
	start_date = begin_date[:4] + "/" + begin_date[4:6] + "/" + begin_date[6:]
	end_date = start_date = end_date[:4] + "/" + end_date[4:6] + "/" + end_date[6:]
	#print(start_date)
	
	config = twint.Config()
	config.Search = sys.argv[1]
	config.Lang = "en"
	config.Limit = sys.argv[2]
	config.Output = "Output.csv"
	config.Since = start_date
	config.Until = end_date

	limit = int(sys.argv[2])
	limit1 = math.ceil(limit * 1.3)
	# # lang = 'english'
	# #tweets = query_tweets(sys.argv[1])
	# #tweets = query_tweets(sys.argv[1], begindate = begin_date.date(), enddate = end_date.date(), limit = limit1, lang = lang)
	# #tweets = twint.run.Search(config)
	# #print(tweets)
	df = pd.DataFrame(columns = ['text'])
	df['text'] = pd.read_csv("Output.csv", error_bad_lines=False)
	# #print(df1.columns.tolist())
	# # df = pd.DataFrame(columns = ['text'])
	# # for i in df1:
	# # 	print("Hello ", i)
	# # 	df.append({'text' : i[:20]}, ignore_index=True)
	# #print(df1)
	# # df = pd.DataFrame(t.__dict__ for t in tweets)
	for index, row in df.iterrows():
		row['text'] = row['text'][45:]
		# print("\n********************************\n")

	df['lang'] = df['text'].apply(lambda x:detector(x))
	# print(df.head)
	df = df[df['lang'] == 'en']

	sentiment = df['text'].apply(lambda x: analyzer.polarity_scores(x))

	df = pd.concat([df, sentiment.apply(pd.Series)], 1)
	df.drop_duplicates(subset = 'text', inplace = True)

	score = df['compound']
	

	np_hist = np.array(score)
	if len(np_hist) > int(limit):
		np_hist = np_hist[:int(limit)]
	hist,bin_edges = np.histogram(np_hist)
	print(bin_edges)

	plt.figure(figsize=[10,8])

	plt.bar(bin_edges[:-1], hist, width = 0.1,color='#0504aa',alpha=0.7)
	plt.xlim(min(bin_edges), max(bin_edges))
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Compound Score',fontsize=15)
	plt.ylabel('Total Tweets',fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title('Sentimental Intensity Distribution ' + sys.argv[1] + ' for ' + str(len(np_hist)) + ' tweets',fontsize=15)
	plt.savefig('plot.png')
	print(program_sentiment(sys.argv[1]))
    
    







