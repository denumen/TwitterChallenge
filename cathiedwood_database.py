import snscrape.modules.twitter as sntwitter
import pandas as pd
import pytz
from datetime import datetime, timedelta
import time


def get_data(time_since):
# Define search query to recieve all tweets to CathieDWood account
    cathiedwood_query = 'to:cathiedwood filter:replies'
    # Retrieve tweets
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(cathiedwood_query).get_items()):
        if tweet.date < time_since: # Stop when tweets are older than time window
            break
        tweets.append([tweet.date, tweet.user.username, tweet.user, tweet.user.displayname, tweet.user.created, tweet.user.verified,
         tweet.user.followersCount, tweet.user.friendsCount, tweet.likeCount, tweet.quoteCount, tweet.replyCount, tweet.retweetCount, tweet.rawContent,
          tweet.inReplyToUser, tweet.conversationId, tweet.id])
    # Store tweets in a dataframe
    df_tweets = pd.DataFrame(tweets, columns=['Datetime', 'username', 'user link', 'display name', 'user created date', 'is verified',
        'followers count', 'friends count', 'likes', 'quotes', 'replies', 'retweets', 'Tweet', 'In Reply To', 'conversationId', 'id'])
    # Define search query to recieve all tweets from CathieDWood account
    cathiedwood_query = 'from:cathiedwood'
    # Retrieve tweets
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(cathiedwood_query).get_items()):
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
    # df_tweets_base has tweets in reply to CathieDWood
    # df_tweets_owner_base has tweets from CathieDWood
    if First_run == True :
        tz = pytz.timezone('Europe/Moscow')  # Set timezone to UTC
        time_since = datetime(2023, 2, 1, tzinfo=tz)
        df_tweets_base, df_tweets_owner_base = get_data(time_since)
        df_tweets_base.to_csv("~/cathiedwood_base.csv", index=False)
        df_tweets_owner_base.to_csv("~/cathiedwood_owner_base.csv", index=False)
        print(First_run)
        First_run = False
        time.sleep(400)
    else:
        # After the first run we get new data every 10minutes 
        # Setting intervals helps us to prevend IP blocking and gives us time to store and process the older data
        time_since = datetime.now(tz) - timedelta(minutes=10)
        df_tweets, df_tweets_owner = get_data(time_since)
        # cathiedwood_all has the old data and new data in reply to CathieDWood
        # cathiedwood_owner_all has the old data and new data from CathieDWood
        cathiedwood_all = pd.concat([df_tweets, df_tweets_base])
        cathiedwood_owner_all = pd.concat([df_tweets_owner, df_tweets_owner_base])
        cathiedwood_all = cathiedwood_all.reset_index(drop=True)
        cathiedwood_owner_all = cathiedwood_owner_all.reset_index(drop=True)
        # Droppping duplicates helps us to prevend saving duplicate data and process them because of scraper bugs or some other issues
        cathiedwood_all = cathiedwood_all.drop_duplicates(subset=['id'])
        cathiedwood_owner_all = cathiedwood_owner_all.drop_duplicates(subset=['id'])
        # we replace the old data with new one in each run, so we can concat new data in next runs and don't miss any data
        df_tweets_base = cathiedwood_all.copy()
        df_tweets_owner_base = cathiedwood_owner_all.copy()
        # Saving data to process them in process_tweets.py
        cathiedwood_all.to_csv("~/cathiedwood_all.csv", index=False)
        cathiedwood_owner_all.to_csv("~/cathiedwood_owner_all.csv", index=False)
        # Print debug output
        #print('hi')
        # time.sleep keep the code waiting for 600seconds before running the next loop 
        time.sleep(600)