import openai
import yaml
import pandas as pd
import os.path
from random import randrange
import numpy as np

import sys
sys.path.append('../classifier')  # add the parent directory to the Python path
from classifier import get_vanilla_dataset, GENRE_DECODINGS

def ask_davinci(question, temp):

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
        temperature=temp,
    )
    
    # Get the response text
    responses = [r["text"] for r in response["choices"]]
    
    return responses

if __name__ == "__main__":
    lyrics = "well i got my first truck when i was three drove a hundred thousand miles on my knees hauled marbles and rocks and thought twice before i hauled a barbie doll bed for the girl next door she tried to pay me with a kiss and i began to understand there s something women like about a pickup man when i turned sixteen i saved a few hundred bucks my first car was a pickup truck i was cruising the town and the first girl i seen was bobbie jo gentry the homecoming queen she flagged me down and climbed up in the cab and said i never knew you were a pickup man you can set my truck on fire and roll it down a hill and i still wouldn t trade it for a coupe de ville i got an eight foot bed that never has to be made you know if it weren t for trucks we wouldn t have tailgates i met all my wives in traffic jams there s just something women like about a pickup man most friday nights i can be found in the bed of my truck on an old chaise lounge backed into my spot at the drive in show you know a cargo light gives off a romantic glow i never have to wait in line at the popcorn stand cause there s something women like about a pickup man you can set my truck on fire and roll it down a hill and i still wouldn t trade it for a coupe de ville i got an eight foot bed that never has to be made you know if it weren t for trucks we wouldn t have tailgates i met all my wives in traffic jams there s just something women like about a pickup man a bucket of rust or a brand new machine once around the block and you ll know what i mean  you can set my truck on fire and roll it down a hill and i still wouldn t trade it for a coupe de ville i got an eight foot bed that never has to be made you know if it weren t for trucks we wouldn t have tailgates i met all my wives in traffic jams there s just something women like about a pickup man yeah there s something women like about a pickup man"
    question = "Write a song in the country genre, that is written in the same style as this song: \n" + lyrics
    temps = [i/10 for i in range(0, 11)]
    for temp in temps:
        responses = ask_davinci(question, temp)
        print("\n\n\ntemp: ", temp)
        print(responses[0])