import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import nltk

#-----Function for cleaning text--------------------------------------
def text_process(mess):
    import string
    from nltk.corpus import stopwords
    #Takes in a string of text, then performs the following:
    #1. Remove all punctuation
    #2. Remove all stopwords
    #3. Returns a list of the cleaned text

    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#---------------------------------------------------------------------------------------------------

#---------------READ DATASET-----------------
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#-------------DATASET DESCRIPTION-----------
print(dataset.head(5))
print("\n")
#print(dataset.describe())
#print("\n")
print(dataset.groupby('author').describe())
print("\n")

#----------FEATURE EXTRACTION--------------
dataset['length'] = dataset['text'].apply(len)
print(dataset.head(10))
#print(dataset.length.describe()) 
#----------length visualization-------
#dataset['length'].plot(bins=100, kind='hist')
#plt.show()
#dataset.hist(column='length', by='author', bins=100,figsize=(12,4))
#plt.show()
#length doesnt seem to be a good feature
#---------------------------------

print(dataset['text'].head(5).apply(text_process))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

bow_transformer = CountVectorizer(analyzer=text_process).fit(dataset['text'])
#print(len(bow_transformer.vocabulary_))
text_bow = bow_transformer.transform(dataset['text'])
tfidf_transformer = TfidfTransformer().fit(text_bow)
text_tfidf = tfidf_transformer.transform(text_bow)
print(text_tfidf.shape)


#---------Machine Learning Model---------------

from sklearn.naive_bayes import MultinomialNB

#spooky_model = MultinomialNB().fit(text_tfidf, dataset['author'])
#all_predictions = spooky_model.predict(text_tfidf)

#model = MultinomialNB()


from sklearn.model_selection import train_test_split
text_train, text_test, target_train, target_test = train_test_split(dataset['text'], dataset['author'], test_size=0.2)

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

#pipeline.fit(text_train,target_train)
#predictions = pipeline.predict(text_test)
#print(classification_report(predictions,target_test))

#---------using files provided by kaggle for train and test--------------
pipeline.fit(dataset['text'],dataset['author'])
predictions = pipeline.predict(test['text'])


prob = pipeline.predict_proba(test['text'])
probabilities = pd.DataFrame(prob)
#pred.to_csv(path_or_buf= 'C:\\Users\\Lycaon\\Documents\\CIC\Mercedes Benz\\', sep=',')
probabilities.to_csv('predictions2.csv', sep=',')
