from os.path import dirname, join
import json
import pandas as pd
from bokeh.io import show, curdoc, output_file
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, Select, Div, ColumnDataSource, CustomJS, DataTable,
                          NumberFormatter, RangeSlider, TableColumn)
from bokeh.models.widgets import DataTable, TableColumn, Panel, Tabs
from bokeh.layouts import row, gridplot
from bokeh.plotting import figure, show, output_file


# Audience data
df = pd.read_csv('~/au_accounts.csv')
source = ColumnDataSource(data=df)
# Accounts data
accounts = pd.read_csv("/root/accounts.csv")
accounts['Sentiment Score'] = accounts['Sentiment Score'].round(3)
source2 = ColumnDataSource(data=accounts)
# Tweets data
tweets_df = pd.read_csv("~/tweets_df.csv")
# format the large_numbers column as strings because JS can't handle it and return 0 for last digits 
tweets_df['id'] = tweets_df['id'].astype(str)
tweets_df['conversationId'] = tweets_df['conversationId'].astype(str)
source3 = ColumnDataSource(data=tweets_df)
# Sentiments of threads and replies
df_sentiments = pd.read_csv("~/df_sentiments.csv")
# format the large_numbers column as strings because JS can't handle it and return 0 for last digits 
df_sentiments['conversationId'] = df_sentiments['conversationId'].astype(str)
source4 = ColumnDataSource(data=df_sentiments)

# define the update function
def update_data():
    # update the audience data
    df = pd.read_csv('~/au_accounts.csv')
    source.data = df
    # update the accounts data
    accounts = pd.read_csv("/root/accounts.csv")
    accounts['Sentiment Score'] = accounts['Sentiment Score'].round(3)
    source2.data = accounts
    # update the tweets data
    tweets_df = pd.read_csv("~/tweets_df.csv")
    # format the large_numbers column as strings because JS can't handle it and return 0 for last digits 
    tweets_df['id'] = tweets_df['id'].astype(str)
    tweets_df['conversationId'] = tweets_df['conversationId'].astype(str)
    source3.data = tweets_df
    # update the sentiments data
    df_sentiments = pd.read_csv("~/df_sentiments.csv")
    # format the large_numbers column as strings because JS can't handle it and return 0 for last digits 
    df_sentiments['conversationId'] = df_sentiments['conversationId'].astype(str)
    source4.data = df_sentiments
# Update function for account select widget
def update(attrname, old, new):
    current = df[df['account'] == select.value]
    #current = df[df['account'] == str(select)]
    print(select)
    source.data = {
        'account'             : current['account'],
        'replies'             : current['Number of replies'],
        'username'             : current['username'],
        'sentiment'             : current['sentiment'],
        'name'             : current['display name'],
        'userlink'             : current['user link'],
        'is_verified'           : current['is verified'],
        'followers_count' : current['followers count'],
        'friends_count' : current['friends count'],
    }
#dropdown = Dropdown(label="Dropdown button", button_type="warning", menu=menu)
#dropdown.js_on_event("menu_item_click", CustomJS(code="console.log('dropdown: ' + this.item, this.toString())"))
select = Select(title="Account:", value="cathiedwood", options=["cathiedwood", "taylorlorenz", "ylecun"])
select.on_change('value', update)
update('value', None, "cathiedwood")  # call update function for initial value
#button = Button(label="Download", button_type="success")
#button.js_on_event("button_click", CustomJS(args=dict(source=source),
#                            code=open("/root/Audience/download.js").read()))
columns = [
    TableColumn(field="account", title="Tracked Account"),
    TableColumn(field="replies", title="Number of Replies"),
    TableColumn(field="username", title="Username"),
    TableColumn(field="sentiment", title="Sentiment"),
    TableColumn(field="name", title="Display Name"),
    TableColumn(field="userlink", title="User Link"),
    TableColumn(field="is_verified", title="is Verified"),
    TableColumn(field="followers_count", title="Followers Count"),
    TableColumn(field="friends_count", title="Friends Count")
]
data_table = DataTable(source=source, columns=columns, width=1100)

# Accounts
columns2 = [
    TableColumn(field="username", title="Username"),\
    TableColumn(field="display name", title="Display Name"),
    TableColumn(field="Sentiment Score", title="Sentiment Score"),    
    TableColumn(field="user created date", title="Created Date"),
    TableColumn(field="is verified", title="Is Verified"),
    TableColumn(field="followers count", title="Followers"),
    TableColumn(field="friends count", title="Friends"),
    TableColumn(field="user link", title="User Link")
]
data_table2 = DataTable(source=source2, columns=columns2, width=1100)

# Tweets
# Update function for tweets select widget
def update3(attrname, old, new):
    current3 = tweets_df[tweets_df['In Reply To'] == 'https://twitter.com/{}'.format(select3.value)]
    #current = df[df['account'] == str(select)]
    print(select3)
    source3.data = {
        'Datetime'             : current3['Datetime'],
        'Tweet'             : current3['Tweet'],
        'sentiment'             : current3['sentiment'],
        'username'             : current3['username'],
        'name'             : current3['display name'],
        'id'             : current3['id'],
        'conversationId'           : current3['conversationId'],
        'is_verified' : current3['is verified'],
        'In Reply To' : current3['In Reply To'],
    }
select3 = Select(title="In Reply To:", value="TaylorLorenz", options=["CathieDWood", "TaylorLorenz", "ylecun"])
select3.on_change('value', update3)
update3('value', None, "TaylorLorenz")  # call update function for initial value
columns3 = [
    TableColumn(field="Datetime", title="Tweet Datetime"),
    TableColumn(field="Tweet", title="Tracked Tweet"),
    TableColumn(field="sentiment", title="sentiment Tweet"),

    TableColumn(field="username", title="Username"),
    TableColumn(field="name", title="Display Name"),
    TableColumn(field="id", title="Tweet id"),
    TableColumn(field="conversationId", title="Conversation ID"),
    TableColumn(field="is_verified", title="is Verified"),
    TableColumn(field="In Reply To", title="In Reply To")
]
data_table3 = DataTable(source=source3, columns=columns3, width=1100)

# Define the JavaScript callback for downloading custom dataset
callback = CustomJS(args=dict(url='http://89.23.110.199/export_data.csv'), code="""
  const link = document.createElement('a');
  link.href = url;
  link.download = 'demodata.csv';
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
""")

# Define functions to create custom dataset using the botton
#  
def export_data_tweets():
    #data = data_table.to_df() # convert the Bokeh data table to a Pandas DataFrame
    data = pd.DataFrame(data_table3.source.data)
    data.to_csv("/var/www/html/export_data.csv", index=False) # export the data to a CSV file
def export_data_audience():
    #data = data_table.to_df() # convert the Bokeh data table to a Pandas DataFrame
    data = pd.DataFrame(data_table.source.data)
    data.to_csv("/var/www/html/export_data.csv", index=False) # export the data to a CSV file
def export_data_accounts():
    #data = data_table.to_df() # convert the Bokeh data table to a Pandas DataFrame
    data = pd.DataFrame(data_table2.source.data)
    data.to_csv("/var/www/html/export_data.csv", index=False) # export the data to a CSV file
def export_data_sentiments():
    #data = data_table.to_df() # convert the Bokeh data table to a Pandas DataFrame
    data = pd.DataFrame(data_table4.source.data)
    data.to_csv("/var/www/html/export_data.csv", index=False) # export the data to a CSV file

# Define bottons for each tab
download_button_tweets = Button(label="Download Data")
download_button_tweets.js_on_click(callback)
download_button_tweets.on_click(export_data_tweets)

download_button_accounts = Button(label="Download Data")
download_button_accounts.js_on_click(callback)
download_button_accounts.on_click(export_data_accounts)

download_button_audience = Button(label="Download Data")
download_button_audience.js_on_click(callback)
download_button_audience.on_click(export_data_audience)

controls = column(select, download_button_audience)
controls2 = column(download_button_accounts)
controls3 = column(select3, download_button_tweets)

# Sentiment
def update4(attrname, old, new):
    current4 = df_sentiments[df_sentiments['username'] == select4.value]
    #current = df[df['account'] == str(select)]
    print(select4)
    source4.data = {
        'username'             : current4['username'],
        'conversationId'             : current4['conversationId'],
        'sentiment_thread'             : current4['sentiment_thread'],
        'sentiment_replies'             : current4['sentiment_replies'],
    }

select4 = Select(title="username:", value="CathieDWood", options=["CathieDWood", "TaylorLorenz", "ylecun"])
select4.on_change('value', update4)
update('value', None, "CathieDWood")  # call update function for initial value
columns4 = [
    TableColumn(field="username", title="Tracked Account"),
    TableColumn(field="conversationId", title="Thread ID"),
    TableColumn(field="sentiment_thread", title="Thread Sentiment"),
    TableColumn(field="sentiment_replies", title="Replies Sentiment"),
]
data_table4 = DataTable(source=source4, columns=columns4, width=1100)
# Save button with custom JS download functionality
download_button_Sentiment = Button(label="Download Data")
download_button_Sentiment.js_on_click(callback)
download_button_Sentiment.on_click(export_data_sentiments)
controls4 = column(select4, download_button_Sentiment)

# Summary Description tab using HTML bokeh widget and Grid for three tracked accounts
# ylecun
div = Div(text="""<div class="row">
  <div class="column">
  <h2>Yann LeCun</h2>
    <img src="http://89.23.110.199/ylecun.jpg" alt="Forest" style="width:70%">
    <p>
    Yann LeCun is a renowned computer scientist and the Director of AI Research at Facebook. He is an expert in the field of artificial intelligence and regularly shares his knowledge with his 434,596 Twitter followers and 601 friends. Yann's tweets cover a range of topics related to AI, including machine learning and deep learning, and he keeps his followers up-to-date with the latest trends and developments in the field.
<br>
Yann is known for his candid opinions and has a large following on Twitter. He regularly hosts Q&A sessions and invites questions on AI and creativity. Yann's keynote speeches at conferences are highly regarded, and his technical insights and humorous observations on Twitter are closely followed by others in the AI community. Overall, Yann LeCun is a respected figure in the field of AI, and his contributions to the community are highly valued.
    </p>
  </div>
</div>""",
width=50, height=100)

# TaylorLorenz
div2 = Div(text="""<div class="row">
  <div class="column">
  <h2>Taylor Lorenz</h2>
    <img src="http://89.23.110.199/taylor.jpg" alt="Forest" style="width:70%">
    <p>
    Taylor Lorenz is a journalist whose writing focuses on social media trends and the impact of technology on culture. She has a verified Twitter account with over 350,000 followers, and she has been active on the platform since 2010. Currently, Lorenz works as a staff writer for The New York Times.
<br>
Lorenz's tweets are engaging and cover a wide range of topics, from fashion to ethics in media. She is not afraid to express her opinions on controversial issues, and her followers appreciate her authenticity and honesty. Her unique perspective and thought-provoking tweets have earned her a reputation as a trusted voice in both social media and journalism.
    </p>
  </div>
</div>""",
width=100, height=100)

# CathieDWood
div3 = Div(text="""
<html>
<head>
<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
h2 {text-align: center;}
p {text-align: left;}
</style>
</head>
<body>
<h2>Cathie Wood</h2>
<img src="http://89.23.110.199/cathie.jpg" alt="Paris" style="width:70%">
<p>
Cathie Wood is a well-known finance and investment expert. She is the founder of ARK Invest, a company that invests in innovative technology companies. Cathie has a significant following on Twitter, where she shares her investment strategies and market insights. Her followers look up to her for advice on how to invest in the financial market, especially in emerging technologies that can transform industries.
<br>
Through her Twitter account, Cathie shares her knowledge on finance and technology trends. She concentrates on emerging technologies that have the potential to revolutionize industries, like digital TV advertising. Additionally, she discusses the impact of innovation on economic theories such as Keynesian economics. Due to her large Twitter following, Cathie is a valuable source of information for investors and industry professionals alike.
</p>
</body>
</html>
""",
width=150, height=70)

grid = gridplot([div, div2, div3], ncols=3, width=400, height=250, toolbar_location='below')

# define rows in each tab
l = row(controls, data_table)
n = row(controls2, data_table2)
m = row(controls3, data_table3)
o = row(div, div2, div3)
p = row(controls4, data_table4)

# define tabs
tab = Panel(child=l, title='Audience')
tab2 = Panel(child=n, title='Accounts')
tab3 = Panel(child=m, title='Tweets')
tab4 = Panel(child=grid, title='Description')
tab5 = Panel(child=p, title='Sentiment')

tabs = Tabs(tabs=[tab4, tab, tab2, tab3, tab5])
# Update data every 3 minutes
#update_interval = 3 * 60 * 1000 # milliseconds
curdoc().add_periodic_callback(update_data, 3*60*1000) # 3 minutes = 3*60*1000 milliseconds

curdoc().add_root(tabs)
curdoc().title = "Data Dashboard"

