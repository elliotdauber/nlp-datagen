import openai
import yaml
import pandas as pd
import os.path

# api ref: https://platform.openai.com/docs/api-reference/edits/create
def ask_davinci(question):

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
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    # Get the response text
    response_text = response["choices"][0]["text"]
    
    return response_text

def generate_song(genre):
    question = "Write a new song in the " + genre + " genre that you haven't generated before. Ensure that the text you generate has no commas in it."
    return ask_davinci(question)

def generate_songs(genre, num_songs, outfile):
    if os.path.isfile(outfile):
        # If file exists, read it into a dataframe
        df = pd.read_csv(outfile)
    else:
        # If file does not exist, create a new dataframe
        df = pd.DataFrame(columns=['Lyric', 'Genre'])

    for i in range(num_songs):
        lyrics = generate_song(genre)
        df = df.append([lyrics, genre])

    # Write the dataframe to a file
    df.to_csv(outfile, index=False)

generate_songs("rock", 2, "generated_data/rock.csv")


