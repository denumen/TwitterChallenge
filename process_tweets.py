import time
import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

def tokenization(data, **kwargs):
    # Tokenizes the input data using a specified tokenizer.
    # **kwargs allows for additional optional arguments to be passed to the function.
    # padding: specifies the padding strategy, with the default being the longest sequence.
    # max_length: specifies the maximum length of the tokenized sequence, with the default being 55.
    # truncation: specifies whether or not to truncate the input sequence if it exceeds max_length.
    # return_tensors: specifies the format of the output, with the default being a TensorFlow tensor.
    return tokenizer(data, 
                   padding=kwargs.get('padding','longest'), 
                   max_length=kwargs.get('max_length',55),
                   truncation=True, 
                   return_tensors="tf")
# Load MobileBERT checkpoint
checkpoint = "google/mobilebert-uncased"
# Load tokenizer for MobileBERT model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load MobileBERT model with hidden states output
model = TFAutoModel.from_pretrained(checkpoint, output_hidden_states=True)
clear_output()

def get_model(**kwargs):
    global model
    # Set default value for max_seq_length if not provided
    max_seq_length = kwargs.get('max_seq_length',55)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
    # Define model inputs
    input_ids = tf.keras.Input(shape=(max_seq_length,), dtype='int32', name='input_ids')
    attention_mask = tf.keras.Input(shape=(max_seq_length,), dtype='int32', name='attention_mask')

    # Tokenize inputs and pass them through the MobileBERT model
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    outputs = model(inputs)
    pooler_output = outputs['pooler_output']
    # Model Head
    # Add a dense layer with 128 hidden units and ReLU activation
    h1 = tf.keras.layers.Dense(128, activation='relu')(pooler_output)
    # Add a dropout layer with rate of 0.2 to prevent overfitting
    dropout = tf.keras.layers.Dropout(0.2)(h1)
    # Add a dense layer with 1 output unit and sigmoid activation because it has only two classes
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    # Create and compile the new model
    new_model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    # Use Adam optimizer with learning rate of 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # Use binary cross-entropy loss for binary classification problem
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # Use binary accuracy as evaluation metric
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return new_model

new_model = get_model()
# Fine-tuning of MobileBert on sentiment140 Dataset code is available in github and kaggle.com/code/sharifi76/fine-tuning-mobile-bert
# Load fine-tuned weights 
new_model.load_weights('/root/sentiment_weights_MobileBert_final.h5')
# this function gets the df and extract tweets as a list to tokenize and make predictions
def sentiments(df):
        df_tweets = df['Tweet']
        df_tweets = df_tweets.to_list()
        inputs = tokenization(df_tweets)
        result_proba = new_model.predict([inputs.input_ids, inputs.attention_mask])
        #df['sentiment'] = result_proba
        return result_proba
# this function has 7 inputs, if first_run is true this function will predict all threads and replies 
# if it's not the first time this code runs we need to give two inputs as base prediction list and lenght of new and old datasets
def sentiment(first_run, a_ind_sent, b_ind_sent, a, a_old, b, b_old):
    print(first_run, a, b, a_old, b_old)
    # predict sentiments for all data
    if first_run == True:
        b_ind_sent = sentiments(df_all_acc)
        a_ind_sent = sentiments(replies)
        return a_ind_sent, b_ind_sent
    # if lenght of replies are not the same as before but lenght of threads are the same
    # model just predict difference index between new and old replies data
    elif b != b_old and a == a_old:
        if b >  b_old:
            ind = b - b_old
            # ind+20 is to prevent bugs because if our data is very short tokenized input shape is not 55 as we defined and model can't predict 
            b_ind = df_all_acc.iloc[:ind+20,:]
            b_ind_sent_new = sentiments(b_ind)
            b_ind_sent_new = b_ind_sent_new[:ind]
            b_ind_sent = np.concatenate([b_ind_sent_new, b_ind_sent])
            return a_ind_sent, b_ind_sent
        else:
            return a_ind_sent, b_ind_sent
    # if lenght of threads are not the same as before but lenght of replies are the same
    # model just predict difference index between new and old threads data
    elif a != a_old and b == b_old:
        if a > a_old:
            ind = a - a_old
            a_ind = replies.iloc[:ind+20,:]
            a_ind_sent_new = sentiments(a_ind)
            a_ind_sent_new = a_ind_sent_new[:ind]
            # Concat old sentiments and new sentiments
            a_ind_sent = np.concatenate([a_ind_sent_new, a_ind_sent])
            return a_ind_sent, b_ind_sent
        else:
            return a_ind_sent, b_ind_sent
    elif b != b_old and a != a_old:
    # if lenght of threads are not the same as before and lenght of replies are not the same
    # model predict difference index between new and old threads and replies data 
        # b
        if b > b_old and a > a_old:
            ind = b - b_old
            b_ind = df_all_acc.iloc[:ind+20,:]
            b_ind_sent_new = sentiments(b_ind)
            b_ind_sent_new = b_ind_sent_new[:ind]
            # Concat old sentiments and new sentiments
            b_ind_sent = np.concatenate([b_ind_sent_new, b_ind_sent])
            # a
            ind = a - a_old
            a_ind = replies.iloc[:ind+20,:]
            a_ind_sent_new = sentiments(a_ind)
            a_ind_sent_new = a_ind_sent_new[:ind]
            # Concat old sentiments and new sentiments
            a_ind_sent = np.concatenate([a_ind_sent_new, a_ind_sent])
            return a_ind_sent, b_ind_sent
        else :
            return a_ind_sent, b_ind_sent
    # if lenght of new data is the same as the input model doesn't predict and return the input sentiments as an output
    elif a == a_old and b == b_old:
        return a_ind_sent, b_ind_sent

taylor_b_old = 0
taylor_a_old = 0
taylor_a_ind_sent = 0
taylor_b_ind_sent = 0

ylecun_b_old = 0
ylecun_a_old = 0
ylecun_a_ind_sent = 0
ylecun_b_ind_sent = 0

cathie_b_old = 0
cathie_a_old = 0
cathie_a_ind_sent = 0
cathie_b_ind_sent = 0

first_run=True
while True:
    # import TaylorLorenz datasets and use model to predict sentiments
    df_all = pd.read_csv("~/taylor_all.csv")
    df_all_acc = pd.read_csv("~/taylor_owner_all.csv")
    replies = df_all[df_all['In Reply To'] == 'https://twitter.com/TaylorLorenz']
    taylorlorenz_tweets = pd.concat([df_all_acc, replies])
    taylorlorenz_tweets = taylorlorenz_tweets.reset_index(drop=True)
    taylor_a = len(replies)
    taylor_b = len(df_all_acc)
    taylor_a_ind_sent, taylor_b_ind_sent = sentiment(
        first_run=first_run, a_ind_sent=taylor_a_ind_sent, b_ind_sent=taylor_b_ind_sent, a=taylor_a,
        a_old=taylor_a_old, b=taylor_b, b_old=taylor_b_old)
    taylor_a_old = len(replies)
    taylor_b_old = len(df_all_acc)
    sentiment_list = np.concatenate([taylor_b_ind_sent, taylor_a_ind_sent])
    taylorlorenz_tweets['sentiment'] = sentiment_list

    # import lecun datasets and use model to predict sentiments
    df_all = pd.read_csv("~/ylecun_all.csv")
    df_all_acc = pd.read_csv("~/ylecun_owner_all.csv")
    replies = df_all[df_all['In Reply To'] == 'https://twitter.com/ylecun']
    ylecun_tweets = pd.concat([df_all_acc, replies])
    ylecun_tweets = ylecun_tweets.reset_index(drop=True)
    ylecun_a = len(replies)
    ylecun_b = len(df_all_acc)
    ylecun_a_ind_sent, ylecun_b_ind_sent = sentiment(first_run, ylecun_a_ind_sent, ylecun_b_ind_sent, ylecun_a, ylecun_a_old, ylecun_b, ylecun_b_old)
    ylecun_a_old = len(replies)
    ylecun_b_old = len(df_all_acc)
    sentiment_list = np.concatenate([ylecun_b_ind_sent, ylecun_a_ind_sent])
    ylecun_tweets['sentiment'] = sentiment_list

    # import CathieDWood datasets and use model to predict sentiments
    df_all = pd.read_csv("~/cathiedwood_all.csv")
    df_all_acc = pd.read_csv("~/cathiedwood_owner_all.csv")
    replies = df_all[df_all['In Reply To'] == 'https://twitter.com/CathieDWood']
    cathiedwood_tweets = pd.concat([df_all_acc, replies])
    cathiedwood_tweets = cathiedwood_tweets.reset_index(drop=True)
    cathie_a = len(replies)
    cathie_b = len(df_all_acc)
    cathie_a_ind_sent, cathie_b_ind_sent = sentiment(first_run, cathie_a_ind_sent, cathie_b_ind_sent, cathie_a, cathie_a_old, cathie_b, cathie_b_old)
    cathie_a_old = len(replies)
    cathie_b_old = len(df_all_acc)
    sentiment_list = np.concatenate([cathie_b_ind_sent, cathie_a_ind_sent])
    cathiedwood_tweets['sentiment'] = sentiment_list

    # Audience List
    # for each account we get top20 audience and mean sentiment of audience tweets 
    # ylecun audience
    audience_lecun = ylecun_tweets[['username', 'display name', 'user link', 'is verified', 'friends count', 'followers count']]
    audience_lecun = audience_lecun[~audience_lecun['username'].str.contains('ylecun')]
    au_lecun = audience_lecun.value_counts()[0:20]
    au_lecun = au_lecun.reset_index()
    au_lecun = au_lecun.rename(columns={0: 'Number of replies'})
    au_lecun['account'] = 'ylecun'
    au_users = au_lecun['username'].to_list()
    mean_list = []
    for user in au_users:
        mean = ylecun_tweets[ylecun_tweets['username'] == user]['sentiment'].mean()
        mean_list.append(mean)
    au_lecun['sentiment'] = mean_list

    # TaylorLorenz audience
    audience_taylorlorenz = taylorlorenz_tweets[['username', 'display name', 'user link', 'is verified', 'friends count', 'followers count']]
    audience_taylorlorenz = audience_taylorlorenz[~audience_taylorlorenz['username'].str.contains('TaylorLorenz')]
    au_taylorlorenz = audience_taylorlorenz.value_counts()[0:20]
    au_taylorlorenz = au_taylorlorenz.reset_index()
    au_taylorlorenz = au_taylorlorenz.rename(columns={0: 'Number of replies'})
    au_taylorlorenz['account'] = 'taylorlorenz'
    au_users = au_taylorlorenz['username'].to_list()
    mean_list = []
    for user in au_users:
        mean = taylorlorenz_tweets[taylorlorenz_tweets['username'] == user]['sentiment'].mean()
        mean_list.append(mean)
    au_taylorlorenz['sentiment'] = mean_list

    # CathieDWood audience
    audience_cathiedwood = cathiedwood_tweets[['username', 'display name', 'user link', 'is verified', 'friends count', 'followers count']]
    audience_cathiedwood = audience_cathiedwood[~audience_cathiedwood['username'].str.contains('CathieDWood')]
    au_cathiedwood = audience_cathiedwood.value_counts()[0:20]
    au_cathiedwood = au_cathiedwood.reset_index()
    au_cathiedwood = au_cathiedwood.rename(columns={0: 'Number of replies'})
    au_cathiedwood['account'] = 'cathiedwood'
    au_users = au_cathiedwood['username'].to_list()
    mean_list = []
    for user in au_users:
        mean = cathiedwood_tweets[cathiedwood_tweets['username'] == user]['sentiment'].mean()
        mean_list.append(mean)
    au_cathiedwood['sentiment'] = mean_list

    # Concat  audience of all tracked accounts to use in API server and Dashboard
    au_accounts = pd.concat([au_cathiedwood, au_taylorlorenz, au_lecun])
    au_accounts = au_accounts.reset_index(drop=True)
    au_accounts.to_csv("~/au_accounts.csv", index=False)

    # Concat threads and replies of all tracked accounts to use in API server and dashboard
    api_thread_all = pd.concat([taylorlorenz_tweets, ylecun_tweets, cathiedwood_tweets])
    api_thread_all = api_thread_all.reset_index(drop=True)
    api_thread_all.to_csv("~/tweets_df.csv", index=False)

    # Sentiment of Each Thread and Audience
    # sentiment of each thread is the exact sentiment of the thread 
    # Sentiment of repliers is the mean of all replies sentiments 
    # Sentiment score of account calculated using the weighted metric
    # weight for each tweet is number of writer followers and then normalized the final sentiment by deviding to sum of all followers numbers for each thread
    # by doing this we can lower the value of spam accounts or new accounts that has high chance of being spam accounts
    df_threads = api_thread_all[api_thread_all['username'].isin(['TaylorLorenz', 'ylecun', 'CathieDWood'])]
    df_threads = df_threads[df_threads['conversationId'] == df_threads['id']]
    df_threads = df_threads[['conversationId', 'sentiment', 'username', 'followers count']]
    df_threads.rename({'sentiment': 'sentiment_thread'}, axis=1, inplace=True)
    audience_thread = api_thread_all[~api_thread_all['username'].isin(['TaylorLorenz', 'ylecun', 'CathieDWood'])]
    audience_thread = audience_thread[audience_thread['conversationId'].isin(df_threads['conversationId'])]
    audience_thread = audience_thread[['conversationId', 'sentiment', 'username', 'followers count']]
    audience_follower = audience_thread.groupby('conversationId')['followers count'].sum().reset_index()
    account_followers = df_threads[['conversationId', 'followers count']]
    sum_followers = account_followers.merge(audience_follower, on='conversationId', how='left')
    sum_followers = sum_followers.replace({np.nan: 0})
    sum_followers['sum_followers'] = sum_followers['followers count_x'] + sum_followers['followers count_y']
    sum_followers = sum_followers[['conversationId', 'sum_followers']]
    sum_followers['sum_followers'] = sum_followers['sum_followers'].astype('int64')
    audience_thread = audience_thread.merge(sum_followers, on='conversationId', how='left')
    df_threads = df_threads.merge(sum_followers, on='conversationId', how='left')
    audience_sentiment = audience_thread.groupby('conversationId')['sentiment'].mean().reset_index()
    df_threads['weighted_sent'] = df_threads['sentiment_thread'] * (df_threads['followers count'] / df_threads['sum_followers'] )
    audience_thread['weighted_sent_audience'] = audience_thread['sentiment'] * (audience_thread['followers count'] / audience_thread['sum_followers'] )
    audience_weighted_sentiment = audience_thread.groupby('conversationId')['weighted_sent_audience'].sum().reset_index()
    df_threads = df_threads.merge(audience_weighted_sentiment, on='conversationId', how='left')
    df_threads = df_threads.replace({np.nan: 0})
    df_threads['Sentiment Score'] = df_threads['weighted_sent'] + df_threads['weighted_sent_audience']
    accounts_score = df_threads.groupby('username')['Sentiment Score'].mean().reset_index()
    df_threads = df_threads.merge(audience_sentiment, on='conversationId', how='left')
    df_threads.rename({'sentiment': 'sentiment_replies'}, axis=1, inplace=True)
    df_sentiment = df_threads[['conversationId', 'username', 'sentiment_thread', 'sentiment_replies']]
    df_sentiment = df_sentiment.replace({np.nan: 'No replies to this thread'})
    df_sentiment.to_csv("~/df_sentiments.csv", index=False)

    # Tracked accounts and Sentiment score of each account calculated in sentiment part
    accounts = api_thread_all[api_thread_all['username'].isin(['TaylorLorenz', 'CathieDWood', 'ylecun'])]
    accounts = accounts[['username', 'display name', 'user created date', 'is verified',
            'followers count', 'friends count', 'user link']]
    accounts = accounts.drop_duplicates(subset='username', keep='first')
    accounts = accounts.reset_index(drop=True)
    accounts = accounts.merge(accounts_score, on='username', how='left')
    accounts.to_csv("~/accounts.csv", index=False)

    # After first run variable first_run will be False and model don't predict all tweets everytime 
    first_run = False
    # pause the loop for 5minutes and then process data to always be up to date
    time.sleep(300)