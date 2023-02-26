from classifier import train_classifier, get_vanilla_dataset, GENRE_ENCODINGS
from datasets import Dataset
import pandas as pd

def get_augmented_dataset(filenames):
    dataset = get_vanilla_dataset()
    dataset_df = dataset.to_pandas()

    for filename in filenames:
        generated_df = pd.read_csv(filename)
        generated_df['genre'] = generated_df['genre'].map(GENRE_ENCODINGS)
        dataset_df = dataset_df.append(generated_df, ignore_index=True)

    return Dataset.from_pandas(dataset_df)


if __name__ == "__main__":
    # dataset = get_vanilla_dataset()
    dataset = get_augmented_dataset(["datagen/generated_data/gospel.csv", "datagen/generated_data/metal.csv", "datagen/generated_data/rap.csv"])
    train_classifier(dataset)