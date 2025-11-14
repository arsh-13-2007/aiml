import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import pickle
 

books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

print(users.head())
print(books.head())
print(ratings.head())     

print(books.shape , users.shape , ratings.shape)

print(books.isnull().sum())
print(ratings.isnull().sum())
print(users.isnull().sum())


# print(books.duplicated().sum())
# print(ratings.duplicated().sum())
# print(users.duplicated().sum())
# popularity based recommender system 

rating_with_name = ratings.merge(books , on='ISBN')
  
num_rating_df =rating_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating' : 'num_rating'}, inplace=True )
print( num_rating_df)


avg_rating_df = rating_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

print( avg_rating_df)

popular_df= num_rating_df.merge(avg_rating_df, on='Book-Title')
print( popular_df)

popular_df = popular_df[popular_df['num_rating']>= 250].sort_values('avg_rating' , ascending=False)
print( popular_df.shape)

add_popular_df = popular_df.merge( books  , on='Book-Title').drop_duplicates('Book-Title').drop(columns=['ISBN' ,'Publisher'  , 'Image-URL-S' ,'Image-URL-L' , 'Year-Of-Publication'])
print( add_popular_df.shape)
print( add_popular_df.head())



# collaborative recommender system 


x = rating_with_name.groupby('User-ID').count()['Book-Rating'] > 200
rating_user = x[x].index
print(rating_user)

filter_rating = rating_with_name[rating_with_name['User-ID'].isin(rating_user)]
y = filter_rating.groupby('Book-Title').count()['Book-Rating'] >=50
famous_books = y[y].index

final_rating = filter_rating[filter_rating['Book-Title'].isin(famous_books)]
pt = final_rating.pivot_table(index='Book-Title', columns='User-ID',values='Book-Rating')

pt.fillna(0 , inplace=True)

similarity_score = cosine_similarity(pt)


def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similarity_items = sorted(list(enumerate(similarity_score[index])) , key=lambda x:x[1] , reverse=True)[1:6]
    data=[]
    for i in similarity_items:
        item=[]
        print(pt.index[i[0]])
        temp_df = books[books['Book-Title']==pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)
    return data

recommend('1984')

# print(popular_df['Image-URL-M'][0
print(add_popular_df.head())

pickle.dump(add_popular_df , open('popular.pkl' ,'wb' ))
pickle.dump(pt , open('pt.pkl','wb'))
pickle.dump(books , open('books.pkl','wb'))
pickle.dump(similarity_score , open('similarity_score.pkl','wb'))
