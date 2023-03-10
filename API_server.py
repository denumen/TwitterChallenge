from flask import Flask, jsonify
import pandas as pd
import json
import time
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
#Import Datasets 
# Tracked Accounts
accounts = pd.read_csv("/root/accounts.csv")
accounts['Sentiment Score'] = accounts['Sentiment Score'].round(3)
# Audience Information for each tracked account
au_accounts = pd.read_csv("/root/au_accounts.csv")
df_sentiments = pd.read_csv("~/df_sentiments.csv")
# format the large_numbers column as strings because JS can't handle it and return 0 for last digits 
df_sentiments['conversationId'] = df_sentiments['conversationId'].astype(str)
# create a dictionary to store the conversation threads
conversations = {}
# Function to update datasets each 30seconds
def load_data():
    global conversations, accounts, au_accounts, df_sentiments
    accounts = pd.read_csv("/root/accounts.csv")
    accounts['Sentiment Score'] = accounts['Sentiment Score'].round(3)
    df_sentiments = pd.read_csv("~/df_sentiments.csv")
    df_sentiments['conversationId'] = df_sentiments['conversationId'].astype(str)
    au_accounts = pd.read_csv("/root/au_accounts.csv")
    df = pd.read_csv("~/tweets_df.csv")

    # define a function to update the conversations dictionary
    def update_conversations(row):
        conversation_id = row['conversationId']
        tweet_text = row['Tweet']
        if conversation_id in conversations:
            conversations[conversation_id].append(tweet_text)
        else:
            conversations[conversation_id] = [tweet_text]
    # apply the update_conversations function to each row in the dataframe
    df.apply(update_conversations, axis=1)

    # print some debug information
    #print("Loaded data from JSON file")
    #print("Number of conversations in JSON file:", len(df))
    #print("Number of conversations in dictionary before update:", len(conversations))

    # update the conversations dictionary
    for conversation_id in df['conversationId'].unique():
        if conversation_id not in conversations:
            conversations[conversation_id] = []
    # print some debug information
    #print("Number of conversations in dictionary after update:", len(conversations))
    #    print('true')
    #else:
    #    print('false')

# define a Flask route that returns tracked accounts as a json output
@app.route('/accounts')
def get_accounts():
    Username_list = accounts['username'].tolist()
    name_list = accounts['display name'].tolist()
    sentiment_list = accounts['Sentiment Score'].tolist()
    date_list = accounts['user created date'].tolist()
    verified_list = accounts['is verified'].tolist()
    followers_list = accounts['followers count'].tolist()
    friends_list = accounts['friends count'].tolist()
    account = [{'username': a, 'display name': r, 'Sentiment Score': b, 'user created date': c,
     'Is Verified': d, 'followers count': e, 'friends count': f} for a, r, b, c, d, e, f in zip(Username_list,
      name_list, sentiment_list, date_list, verified_list, followers_list, friends_list)]
    return jsonify(account)

# define a Flask route that returns sentiments of each thread at Audience level and thread level as a json output
@app.route('/sentiment/<thread>', methods=['GET'])
def get_sentiments(thread):
    sentiments_list = df_sentiments[df_sentiments['conversationId'] == thread]
    thread_id = sentiments_list['conversationId'].tolist()
    thread_owner = sentiments_list['username'].tolist()
    thread_sentiment = sentiments_list['sentiment_thread'].tolist()
    audience_sentiment = sentiments_list['sentiment_replies'].tolist()
    result = [{'Thread ID': a, 'Thread owner': r, 'Thread sentiment': b, 'Audience sentiment': c} for a, r, b, c in zip(thread_id,
      thread_owner, thread_sentiment, audience_sentiment)]
    return jsonify(result)

# define a Flask route that returns top20 Audience of each tracked accounts as a json output
@app.route('/audience/<account>', methods=['GET'])
def get_audience(account):
    account_df = au_accounts[au_accounts['account'] == account]
    audience_list = account_df['username'].tolist()
    audience_name = account_df['display name'].tolist()
    audience_link = account_df['user link'].tolist()
    audience_verified = account_df['is verified'].tolist()
    audience_followers = account_df['followers count'].tolist()
    audience_friends = account_df['friends count'].tolist()

    replies_list = account_df['Number of replies'].tolist()
    result = [{'audience username': a, 'number of replies': r, 'Audience Display Name': b, 'Audience User Link': c,
     'Audience Is Verified': d, 'Audience followers count': e, 'friends count': f} for a, r, b, c, d, e, f in zip(audience_list,
      replies_list, audience_name, audience_link, audience_verified, audience_followers, audience_friends)]
    return jsonify(result)

# define a Flask route that returns Tweets and replies for each thread 
@app.route('/tweets/<conversation_id>')
def get_tweets(conversation_id):
    # check if the conversation ID exists in the dictionary
    if int(conversation_id) in conversations:
        # if it does, return the conversation threads as JSON
        return jsonify(conversations[int(conversation_id)])
    else:
        # if it doesn't, return an error message
        return jsonify({'error': 'Conversation ID not found.'}), 404

# define a function to update the conversations dictionary every 30 seconds
def update_conversations_scheduler():
    global conversations
    conversations = {}
    load_data()
    print("Dataset refreshed at", time.strftime('%H:%M:%S'))
# create a scheduler that calls the update_conversations_scheduler function every 30 seconds
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_conversations_scheduler, trigger='interval', seconds=30)
scheduler.start()

# shut down the scheduler when the app exits
atexit.register(lambda: scheduler.shutdown())

# start the app and permit server to run on public IP
if __name__ == '__main__':
    load_data()
    app.run(debug=True, host='0.0.0.0')
