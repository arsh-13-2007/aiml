import numpy as np 
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB , MultinomialNB , BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , accuracy_score , confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam_dataset.csv' ,encoding="latin1")
print( data.head())
print( data.shape)
print(data.isnull().sum())

#     task
# 1. data cleaning 
# 2. eda
# 3. test preprocessing 
# 4. modal built
# 5. evaluation 
# 6. improvement 
# 7. website 
# 8. deploy

data = data.drop(columns=['Unnamed: 2' , 'Unnamed: 3' ,'Unnamed: 4'] , axis= 1 )

data.rename(columns={'v1' : 'target' , 'v2' : 'text'} ,inplace=True )
le = LabelEncoder()
data['target'] = le.fit_transform(data['target'])
print(data.head())

print(data.duplicated().sum())

data = data.drop_duplicates()    # droping   duplicate rows 
print( data.head())
print( data.shape)
 

#                     EDA 

print( data['target'].value_counts())     # by this we able to see that thsi dataset is unbalance dataset 
plt.pie(data['target'].value_counts() , labels= ['ham' ,'spam'] ,autopct="%0.2f")
plt.show()



data['num_character']= data['text'].apply(len)
print( data.head())
# make sure punkt is downloaded



data['num_words'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x)))
data['num_sentences'] = data['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(data.head())

print(data[['num_character' , 'num_words' , 'num_sentences']].describe())

sns.pairplot(data , hue='target')
plt.show()


sns.heatmap(data.corr(numeric_only=True ) , annot=True )
plt.show()       # using headmap to know tghe coreleation between them so we able to select the main main columns 
# after this we select num_character column only because it is higher corr with target 


# data preprocessing 

def transform_text( text):
    text = text.lower()
    text = nltk.word_tokenize( text)  # it  is use to split into single single word 
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y [:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i  in text:
        y.append(ps.stem(i))

    return " ".join(y)


data['transform_text']=data['text'].apply(transform_text)
wc = WordCloud(width= 500 ,height = 500 , min_font_size=10 , background_color='white')
spam_wc = wc.generate(data[data['target']==1]['transform_text'].str.cat(sep=" "))

plt.imshow(spam_wc)
plt.show()

ham_wc = wc.generate(data[data['target']==0]['transform_text'].str.cat(sep=" "))
plt.imshow(ham_wc)
plt.show()


X = cv.fit_transform(data['transform_text']).toarray()
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2 )


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
# gnb.fit(X_train , y_train)
# y_pred = gnb.predict( X_test)


mnb.fit(X_train , y_train)
y_pred = mnb.predict( X_test)


# bnb.fit(X_train , y_train)
# y_pred = bnb.predict( X_test)





accuracy = accuracy_score(y_pred , y_test)
print(accuracy)
print( confusion_matrix(y_pred ,y_test))
print( precision_score( y_pred , y_test))