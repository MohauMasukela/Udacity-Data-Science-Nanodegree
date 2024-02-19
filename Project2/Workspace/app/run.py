import json

import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify


import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data

engine = create_engine('sqlite:///data/DisasterResponse.db')
conn=engine.connect()
pd.set_option('display.max_columns',85)
df = pd.read_sql_table('DisasterResponse', engine)

genre_count=df['genre'].value_counts().reset_index()
classifier=df.iloc[:, 4:].sum().rename("count").reset_index().sort_values(by="count",ascending=False)

#load model
model = joblib.load("models/classifier.pkl")

def create_plot():

    fig = make_subplots(rows=1, cols=2,column_widths=[0.6, 0.4],subplot_titles=("Message by Category", "Message by Source"))

    fig.add_bar(y=classifier['count'], x=classifier['index'], row=1, col=1)
    fig.add_bar(y=genre_count['count'], x=genre_count['genre'], row=1, col=2)


    fig.update_layout(height=600,width=1300,showlegend=False)
 

 
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    plot = create_plot()



    return render_template('master.html', plot=plot)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()