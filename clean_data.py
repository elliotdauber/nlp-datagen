import pandas as pd
import numpy as np
import re

def unique_cols(df):
    for col in df.columns:
        # Get the number of unique values in the column
        unique_values = df[col].nunique()
        print("Column Name:", col)
        print("Unique Values:", unique_values)
    print("\n")

lyrics = pd.read_csv("data/lyrics-data.csv")
artists = pd.read_csv("data/artists-data.csv")
# unique_cols(lyrics)
# unique_cols(artists)

lyrics = lyrics[lyrics['language'].str.contains("en", na=False)].dropna()

df_merged = pd.merge(lyrics[["ALink", "Lyric"]], artists[["Link", "Genres"]], left_on='ALink', right_on="Link", how='left')[["Lyric", "Genres"]]
# unique_cols(df_merged)

df_dedup = df_merged.drop_duplicates(subset='Lyric')
# unique_cols(df_dedup)

# Might not want this, tags might be good for NN to learn song structure
df_dedup['Lyric'] = df_dedup['Lyric'].astype(str).apply(lambda x: re.sub("\[.*?\]", "", x))

genre_df = df_dedup['Genres'].str.split(";", expand=True)
genre_df = genre_df.rename(columns = lambda x : f"Genre{x}")
df_split = pd.concat([df_dedup.drop(['Genres'], axis=1), genre_df], axis=1)
df_split = df_split.drop('Genre1', axis=1)
df_split = df_split.drop('Genre2', axis=1)
df_split = df_split.drop('Genre3', axis=1)
df_split = df_split.rename(columns={'Genre0': 'Genre'})

genres = ["Rap", "Heavy Metal", "Country", "Gospel/Religioso"]
df_split = df_split.loc[(df_split['Genre'].isin(genres))]
df_split = df_split.dropna()

unique_cols(df_split)

value_counts = df_split['Genre'].value_counts().head(20)
print(value_counts)

df_split.to_csv('data/clean-data.csv', index=False)
