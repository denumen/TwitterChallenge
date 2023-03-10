# TwitterChallenge
You can find the challenge description [here](https://equatorial-sternum-35b.notion.site/Twitter-Watch-052f5ae4fd1d440ba7a590af040065e4)

For this challenge, I collected all the tweets and replies from Yann LeCun, Taylor Lorenz, and Cathie Wood since 2023-02-01 and extracted relevant information.

The list of accounts:
- @taylorlorenz
- @cathiedwood
- @ylecun

**API Endpoints:**
- API endpoint is accessa:
    - /accounts: return a json list of all tracked accounts:
    - /tweets/<conversationID> : return a json of the user's conversation threads since start.
    - /audience/<tracked_username> : return a json of information about the audience for a user's account.
    - /sentiment/<threadID> : return a json about the sentiment information of an account (e.g. thread level, audience level)
    
You can use this link to access the API http://twitterchallenge.ir:5000/
```
<threadID> is tweet id of tracked accounts 
<conversationID> is tweetID of tracked accounts and tweetID of parents for replies of tracked accounts.
```
  
## Data Dashboard:
To access the dashboard, please follow [this link](http://twitterchallenge.ir:5006/data)
    
The dashboard was implemented using the Bokeh library, which is a Python package that allows for the creation of interactive visualizations and applications in web browsers. The dashboard consists of several tabs, each of which displays a different set of data.



    
## Source Code:  
  to reproduce the results in the server you can run the following scripts and add them as a service to your server:
```
  run the fine-tuning-mobile-bert.ipynb and save the weights to load them in process_tweets.py file 
  python cathiedwood_database.py 
  python ylecun.py 
  python taylorlorenz.py 
  python process_tweets.py 
  python API_server.py 
  bokeh serve --allow-websocket-origin=* ~/data
```
  ### Sentiment analyze
 For this challenge, we used the BertMobile model as our base model because we needed to handle real-time predictions on streaming data and achieve fast results. BertMobile was a suitable choice since it's affordable and provides decent accuracy after fine-tuning. However, to enhance the model's performance, we needed to fine-tune it on a three-class Twitter dataset, as the original model couldn't predict neutral sentiments accurately. This fine-tuned model can improve the overall performance of the model. You can find the fine-tuning notebook on [this link](https://www.kaggle.com/code/sharifi76/fine-tuning-mobile-bert) or in the repository under the name 'fine-tuning-mobile-bert.ipynb'.
  ### Summary description generator 
You can find the code used for generating descriptions in the "Description.ipynb" notebook. The code utilizes the GPT-3.5-turbo pretrained model to generate descriptions based on the tweets and information extracted from each tracked account. Personal information such as the number of followers and tweets for each account were used to create different prompts for the model. By randomly mixing the generated summaries sequentially, the text generated was more accurate and stable across different runs.
## sentiment metric
To evaluate the threads and replies of the tracked account, I utilized a weighted sentiment approach that considers the impact of spam replies and prioritizes the sentiment of the account owners. This approach provides a more reliable metric, as the account owners' opinions and reactions are essential in determining the overall sentiment of the thread. To give more weightage to the account owners' sentiment, I used a weightage of 2x, which means that their sentiment carried much importance as others in the thread. This approach allows for a more accurate representation of the sentiment and provides valuable insights into the account's online presence.
    
  <br>
<p align="center" style="background-color:white">
  <img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbg_white%20%5Clarge%20%5Cbegin%7Baligned%7D%20%26%5Csum_%7B%5Cmathrm%7Bi%7D%3D1%7D%5Em%5Cleft%28%5Cmathrm%7Bx%7D_%7B%5Cmathrm%7Bi%7D%7D%20*%20%5Cfrac%7B%5Cmathrm%7Bw%7D_%7B%5Cmathrm%7Bi%7D%7D%7D%7B2%20*%5Cmathrm%7Bw%7D_%7B%5Ctext%20%7Bowner%20%7D%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7B%5Cmathrm%7Bm%7D%7D%20w_i%7D%5Cright%29&plus;%5Cleft%28x_%7B%5Ctext%20%7Bowner%20%7D%7D%20*%20%5Cfrac%7B2%5E*%20w_%7B%5Ctext%20%7Bowner%20%7D%7D%7D%7B2%20*%20%5Cmathrm%7Bw%7D_%7B%5Ctext%20%7Bowner%20%7D%7D&plus;%5Csum_%7Bi%3D1%7D%5E%7B%5Cmathrm%7Bm%7D%7D%20w_i%7D%5Cright%29%20%5C%5C%20%26%20x_%7B%5Ctext%20%7Bowner%20%7D%7D%3A%20%5Ctext%20%7B%20sentiment%20score%20of%20tweet%20owner%20%7D%20%5C%5C%20%26%20%5Cmathrm%7Bx%7D_%7B%5Cmathrm%7Bi%7D%7D%20%5Ctext%20%7B%20%3A%20sentiment%20score%20of%20replies%20of%20a%20tweet%20%7D%20%5C%5C%20%26%20%5Cmathrm%7Bw%7D_%7B%5Cmathrm%7Bi%7D%7D%20%5Ctext%20%7B%20%3A%20number%20of%20replier%20followers%20%7D%20%5C%5C%20%26%20w_%7B%5Ctext%20%7Bowner%20%7D%7D%3A%20%5Ctext%20%7B%20number%20of%20owner%20of%20the%20tweet%20followers%20%7D%20%5C%5C%20%26%20%5Cmathrm%7Bm%7D%20%5Ctext%20%7B%20%3A%20number%20of%20replies%20to%20a%20thread%20%7D%20%5C%5C%20%26%20%5Cend%7Baligned%7D" title="your LaTeX formula" />
</p>
  

  
