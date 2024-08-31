#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\solas\Downloads\movie_data\movies.csv")


# In[3]:


df2 = pd.read_csv(r"C:\Users\solas\Downloads\movie_data\ratings.csv")


# In[4]:


df.shape


# In[5]:


df2.shape


# In[6]:


df.head()


# In[7]:


df.tail()


# In[12]:


df['movieId'].unique()


# In[8]:


df2.head()


# In[9]:


df2.tail()


# In[9]:


df2['userId'].unique()


# In[10]:


df2['userId'].nunique()


# In[15]:


merged_df = pd.merge(df,df2, on='movieId')

# Group by 'title' and count the number of ratings
rating_counts = merged_df.groupby('title')['rating'].count()

# Find the movie with the maximum number of ratings
max_rated_movie = rating_counts.idxmax()
max_ratings = rating_counts.max()
print(f"The movie with the maximum number of ratings is: {max_rated_movie} with {max_ratings} ratings.")


# In[16]:


df3 = pd.read_csv(r"C:\Users\solas\Downloads\movie_data\tags.csv")


# In[17]:


df3.head()


# In[18]:


matrix_movie_id = df[df['title'] == "Matrix, The (1999)"]['movieId'].values[0]

# Select all tags associated with this movie
matrix_tags = df3[df3['movieId'] == matrix_movie_id]['tag'].unique()

# Display the tags
print("Tags for 'The Matrix (1999)':", matrix_tags)


# In[19]:


# Find the movie ID for "Terminator 2: Judgment Day (1991)"
terminator_movie_id = df[df['title'] == "Terminator 2: Judgment Day (1991)"]['movieId'].values[0]

# Select all ratings for this movie
terminator_ratings = df2[df2['movieId'] == terminator_movie_id]['rating']

# Calculate the average rating
average_rating = terminator_ratings.mean()

# Display the average rating
print(f"The average user rating for 'Terminator 2: Judgment Day (1991)' is: {average_rating}")


# In[21]:


import pandas as pd

# Load dataset
# Group the user ratings based on 'movie_id' and apply aggregation (count and mean)
grouped_ratings = df2.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    average_rating=('rating', 'mean')
).reset_index()

# Apply an inner join between 'movies_df' and 'grouped_ratings'
merged_df = pd.merge(df, grouped_ratings, on='movieId', how='inner')

# Filter only those movies which have more than 50 user ratings
popular_movies_df = merged_df[merged_df['rating_count'] > 50]

# Find the movie with the highest average rating
most_popular_movie = popular_movies_df.loc[popular_movies_df['average_rating'].idxmax()]

# Display the most popular movie
print(f"The most popular movie based on average user ratings is: '{most_popular_movie['title']}' with an average rating of {most_popular_movie['average_rating']:.2f}.")


# In[22]:


#Group the user ratings based on 'movieId' and apply aggregation (count and mean)
grouped_ratings = df2.groupby('movieId').agg(
   rating_count=('rating', 'count'),
   average_rating=('rating', 'mean')
).reset_index()

# Apply an inner join between 'movies_df' and 'grouped_ratings'
merged_df = pd.merge(df, grouped_ratings, on='movieId', how='inner')

# Sort the movies based on the number of ratings in descending order and select the top 5
top_5_popular_movies = merged_df.sort_values(by='rating_count', ascending=False).head(5)

# Display the top 5 popular movies
print("Top 5 popular movies based on the number of user ratings:")
for index, row in top_5_popular_movies.iterrows():
   print(f"{row['title']} - Number of Ratings: {row['rating_count']}")


# In[31]:


grouped_ratings = df2.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    average_rating=('rating', 'mean')
).reset_index()

# Apply an inner join between 'movies_df' and 'grouped_ratings'
merged_df = pd.merge(df, grouped_ratings, on='movieId', how='inner')

# Filter only movies with more than 50 user ratings
popular_movies_df = merged_df[merged_df['rating_count'] > 50]

# Filter only Sci-Fi movies
sci_fi_movies_df = popular_movies_df[popular_movies_df['title'].str.contains('genres', case=False, na=False)]

# Sort the Sci-Fi movies based on the number of ratings in descending order and select the third most popular
if len(sci_fi_movies_df) >= 3:
    sorted_sci_fi_movies = sci_fi_movies_df.sort_values(by='rating_count', ascending=False)
    third_most_popular_sci_fi = sorted_sci_fi_movies.iloc[2]
    
    # Display the third most popular Sci-Fi movie
    print(f"The third most popular Sci-Fi movie based on the number of user ratings is: '{third_most_popular_sci_fi['title']}' with {third_most_popular_sci_fi['rating_count']} ratings.")
else:
    print("There are fewer than three Sci-Fi movies with more than 50 ratings in the dataset.")


# In[18]:


df4 = pd.read_csv(r"C:\Users\solas\Downloads\movie_data\links.csv")


# In[19]:


df4.head()


# In[20]:




# Merge the two DataFrames to include IMDB IDs with the popular movies
merged_df = pd.merge(df, df4, on='movieId')


# In[ ]:


# Drop rows where the rating is missing
merged_df = merged_df.dropna(subset=['imdbRating'])

# Find the movie with the highest IMDB rating
highest_rating_movie = merged_df.loc[merged_df['imdbRating'].idxmax()]

# Get the movieId of the movie with the highest IMDB rating
highest_imdb_movie_id = highest_rating_movie['movieId']

print(f"The movieId of the movie with the highest IMDB rating is: {highest_imdb_movie_id}")


# In[ ]:


import pandas as pd
import requests

# Load the datasets


# Merge the two datasets to include IMDB IDs with the popular movies
merged_df = pd.merge(df, d4f, on='movieId')

# Define a function to get IMDB rating from OMDB API
def get_imdb_rating(imdb_id, api_key):
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['Response'] == 'True':
        return float(data.get('imdbRating', 0))
    return None

# Fetch IMDB ratings for the movies in the dataset
api_key = 'YOUR_OMDB_API_KEY'  # Replace with your OMDB API key

def fetch_ratings(df, api_key):
    ratings = {}
    for imdb_id in df['imdbId']:
        rating = get_imdb_rating(imdb_id, api_key)
        if rating is not None:
            ratings[imdb_id] = rating
    return ratings

ratings = fetch_ratings(merged_df, api_key)

# Add IMDB ratings to the DataFrame
merged_df['imdbRating'] = merged_df['imdbId'].map(ratings)

# Filter Sci-Fi movies
# Assuming you have a column 'genres' or similar that includes genre information
# If not, you will need a separate dataset or method to identify Sci-Fi movies
sci_fi_movies_df = merged_df[merged_df['genres'].str.contains('Sci-Fi', na=False)]

# Drop rows where the rating is missing
sci_fi_movies_df = sci_fi_movies_df.dropna(subset=['imdbRating'])

# Find the Sci-Fi movie with the highest IMDB rating
highest_rating_sci_fi_movie = sci_fi_movies_df.loc[sci_fi_movies_df['imdbRating'].idxmax()]

# Get the movieId of the Sci-Fi movie with the highest IMDB rating
highest_imdb_sci_fi_movie_id = highest_rating_sci_fi_movie['movieId']

print(f"The movieId of the Sci-Fi movie with the highest IMDB rating is: {highest_imdb_sci_fi_movie_id}")


# In[22]:


import pandas as pd

# Load the dataset

# Search for "Forrest Gump" to handle potential title mismatches
forrest_gump_df = df[df['title'].str.contains("Forrest Gump", na=False, case=False)]

# Check if we have any matching movies
if not forrest_gump_df.empty:
    # Extract the movieId
    forrest_gump_movie_id = forrest_gump_df['movieId'].values[0]
    print(f"The movieId for 'Forrest Gump (1994)' is: {forrest_gump_movie_id}")
else:
    print("Movie 'Forrest Gump (1994)' not found in the dataset.")


# In[26]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0"*n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {'Content-Type': 'text/html; charset=UTF-8', 
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 
                      'Accept-Encoding': 'gzip, deflate, br'}
    response = requests.get(URL, headers=request_header)
    soup = BeautifulSoup(response.text, 'html.parser')
    imdb_rating = soup.find('span', attrs={'itemprop': 'ratingValue'})
    return float(imdb_rating.text) if imdb_rating else np.nan

def highest_rated_movie(movieiId):
    highest_rating = -1
    best_movie_id = None

    for movie_id in movie_ids:
        rating = scrapper(movieId)
        if rating > highest_rating:
            highest_rating = rating
            best_movie_id = movieId

    return best_movie_id


# In[ ]:




