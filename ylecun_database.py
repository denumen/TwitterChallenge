import snscrape.modules.twitter as sntwitter
import pandas as pd
import pytz
from datetime import datetime, timedelta
import time


def get_data(time_since):
# Define search query to recieve all tweets to ylecun account
    ylecun_query = 'to:ylecun filter:replies'
    # Retrieve tweets
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(ylecun_query).get_items()):
        if tweet.date < time_since: # Stop when tweets are older than time window
            break
        tweets.append([tweet.date, tweet.user.username, tweet.user, tweet.user.displayname, tweet.user.created, tweet.user.verified,
         tweet.user.followersCount, tweet.user.friendsCount, tweet.likeCount, tweet.quoteCount, tweet.replyCount, tweet.retweetCount, tweet.rawContent,
          tweet.inReplyToUser, tweet.conversationId, tweet.id])
    # Store tweets in a dataframe
    df_tweets = pd.DataFrame(tweets, columns=['Datetime', 'username', 'user link', 'display name', 'user created date', 'is verified',
        'followers count', 'friends count', 'likes', 'quotes', 'replies', 'retweets', 'Tweet', 'In Reply To', 'conversationId', 'id'])
    # Define search query to recieve all tweets from ylecun account
    ylecun_query = 'from:ylecun'
    # Retrieve tweets
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(ylecun_query).get_items()):
        if tweet.date < time_since: # Stop when tweets are older than time window
            break
        tweets.append([tweet.date, tweet.user.username, tweet.user, tweet.user.displayname, tweet.user.created, tweet.user.verified,
         tweet.user.followersCount, tweet.user.friendsCount, tweet.likeCount, tweet.quoteCount, tweet.replyCount, tweet.retweetCount, tweet.rawContent,
          tweet.inReplyToUser, tweet.conversationId, tweet.id])
    # Store tweets in a dataframe
    df_tweets_owner = pd.DataFrame(tweets, columns=['Datetime', 'username', 'user link', 'display name', 'user created date', 'is verified',
        'followers count', 'friends count', 'likes', 'quotes', 'replies', 'retweets', 'Tweet', 'In Reply To', 'conversationId', 'id'])

    return df_tweets, df_tweets_owner
# define variable for the first run and set to False after the first run 
# first run saves the initialized dataset to update it in  next runs 
First_run = True
while True:
    # If it's the first run we get the dataset since 2023-2-1 untill the time we running the code
    # df_tweets_base has tweets in reply to ylecun
    # df_tweets_owner_base has tweets from ylecun
    if First_run == True :
        tz = pytz.timezone('Europe/Moscow')  # Set timezone to Your server timezone
        time_since = datetime(2023, 2, 1, tzinfo=tz)
        df_tweets_base, df_tweets_owner_base = get_data(time_since)
        df_tweets_base.to_csv("~/ylecun_base.csv", index=False)
        df_tweets_owner_base.to_csv("~/ylecun_owner_base.csv", index=False)
        print(First_run)
        First_run = False
        time.sleep(400)
    else:
        # After the first run we get new data every 10minutes 
        # Setting intervals helps us to prevend IP blocking and gives us time to store and process the older data
        time_since = datetime.now(tz) - timedelta(minutes=10)
        df_tweets, df_tweets_owner = get_data(time_since)
        # ylecun_all has the old data and new data in reply to ylecun
        # ylecun_owner_all has the old data and new data from ylecun
        ylecun_all = pd.concat([df_tweets, df_tweets_base])
        ylecun_owner_all = pd.concat([df_tweets_owner, df_tweets_owner_base])
        ylecun_all = ylecun_all.reset_index(drop=True)
        ylecun_owner_all = ylecun_owner_all.reset_index(drop=True)
        # Droppping duplicates helps us to prevend saving duplicate data and process them because of scraper bugs or some other issues
        ylecun_all = ylecun_all.drop_duplicates(subset=['id'])
        ylecun_owner_all = ylecun_owner_all.drop_duplicates(subset=['id'])
        # we replace the old data with new one in each run, so we can concat new data in next runs and don't miss any data
        df_tweets_base = ylecun_all.copy()
        df_tweets_owner_base = ylecun_owner_all.copy()
        # Saving data to process them in process_tweets.py
        ylecun_all.to_csv("~/ylecun_all.csv", index=False)
        ylecun_owner_all.to_csv("~/ylecun_owner_all.csv", index=False)
        # Print debug output
        #print('hi')
        # time.sleep keep the code waiting for 600seconds before running the next loop 
        time.sleep(600)