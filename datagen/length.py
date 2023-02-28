import pandas as pd
import sys

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    print(len(df.index))