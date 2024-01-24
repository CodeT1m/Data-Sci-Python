from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib




app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World'
    render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('YoutubeSpamMergedData.csv')
    df_data = df[['CONTENT', 'CLASS']]
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_x)
    clf = MultinomialNB()
    clf.fit(X, df_y)

    # Save the trained classifier
    joblib.dump(clf, 'classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    
    return render_template('result.html')




if __name__ == '__main__':
    app.run(debug=True)