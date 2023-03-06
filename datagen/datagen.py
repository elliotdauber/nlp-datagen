import openai
import yaml
import pandas as pd
import os.path
from random import randrange
import numpy as np

import sys
sys.path.append('../classifier')  # add the parent directory to the Python path
from classifier import get_vanilla_dataset, GENRE_DECODINGS

# api ref: https://platform.openai.com/docs/api-reference/edits/create
def ask_davinci(question, num_responses=1):

    with open("secrets.yaml") as file:
        data = yaml.safe_load(file)
        # Set the API key
        openai.api_key = data['open-ai']
    
    # Define the prompt
    prompt = (f"{question}")
    
    # Use the OpenAI API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=randrange(750, 1024),
        n=num_responses,
        stop=None,
        temperature=0.9,
    )
    
    # Get the response text
    responses = [r["text"] for r in response["choices"]]
    
    return responses

def generate_song(genre, n):
    question = "Write a new song in the " + genre + " genre that you haven't generated before. Ensure that the text you generate has no commas in it."
    return ask_davinci(question, n)

def generate_songs(genre, num_songs, outfile):
    if os.path.isfile(outfile):
        # If file exists, read it into a dataframe
        df = pd.read_csv(outfile)
    else:
        # If file does not exist, create a new dataframe
        df = pd.DataFrame(columns=['lyrics', 'genre'])

    print(df)
    for i in range(num_songs):
        responses = generate_song(genre, 10)
        print("generated number", i)
        for r in responses:
            new_df = {'lyrics': r, 'genre': genre}
            df = df.append(new_df, ignore_index = True)

    # Write the dataframe to a file
    df.to_csv(outfile, index=False)

def generate_songs_based_on_others(n_per, genre_to_generate=None):
    dataset = get_vanilla_dataset()
    df = pd.DataFrame(dataset)
    grouped = df.groupby('genre')
    sampled = grouped.apply(lambda x: x.sample(n=n_per, random_state=np.random.seed()))
    sampled = sampled.reset_index(drop=True)

    for name, group in sampled.groupby('genre'):
        genre = GENRE_DECODINGS[name]
        if genre_to_generate is None or genre_to_generate == genre:
            print(genre)
            outfile = "generated_data/" + genre + ".csv"
            if os.path.isfile(outfile):
                # If file exists, read it into a dataframe
                df = pd.read_csv(outfile)
            else:
                # If file does not exist, create a new dataframe
                df = pd.DataFrame(columns=['lyrics', 'genre'])
            # do something with each group
            print(group.head())
            for index, row in group.iterrows():
                lyrics = row['lyrics']
                question = "Write a song in the " + genre + " genre, that is written in the same style as this song: \n" + lyrics
                responses = ask_davinci(question, 2)
                for r in responses:
                    new_df = {'lyrics': r, 'genre': genre}
                    df = df.append(new_df, ignore_index = True)

            df.to_csv(outfile, index=False)


# generate_songs("rap", 4, "generated_data/rap.csv")

if __name__ == "__main__":
    # example: python3 datagen.py 10 rap
    n_per = int(sys.argv[1])
    genre = sys.argv[2] if len(sys.argv) > 2 else None
    generate_songs_based_on_others(n_per, genre)


