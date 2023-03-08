import pandas as pd
import sys

if __name__ == "__main__":
    genres = ["country", "gospel", "metal", "rap"]
    for genre in genres:
        df = pd.read_csv("generated_data/" + genre + ".csv")
        print(genre, len(df.index))