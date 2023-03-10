from classifier import train_classifier, get_vanilla_dataset, sample_from_dataset, GENRE_ENCODINGS
from datasets import Dataset, concatenate_datasets
from experiments import EXPERIMENTS
import pandas as pd
import sys

def augment_dataset(dataset, filenames):
    dataset_df = dataset.to_pandas()

    for filename in filenames:
        generated_df = pd.read_csv(filename)
        generated_df['genre'] = generated_df['genre'].map(GENRE_ENCODINGS)
        dataset_df = dataset_df.append(generated_df, ignore_index=True)

    return Dataset.from_pandas(dataset_df)


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else None

    vanilla_dataset = get_vanilla_dataset()

    for e in EXPERIMENTS:
        if experiment_name is None or e.name == experiment_name:
            print("gathering dataset for experiment " + experiment_name)
            e.show()

            real_props = e.real
            generated_props = e.generated
            
            combined_dataset = Dataset.from_dict({"genre": [], "lyrics": []})
            # add the real data
            for genre, num_examples in real_props.items():
                if num_examples > 0:
                    sub_dataset = sample_from_dataset(vanilla_dataset, genre, num_examples)
                    combined_dataset = concatenate_datasets([combined_dataset, sub_dataset])

            # add the generated data
            for genre, num_examples in generated_props.items():
                if num_examples > 0:
                    filename = "../datagen/generated_data/" + genre + ".csv"
                    generated_df = pd.read_csv(filename)
                    generated_df['genre'] = generated_df['genre'].map(GENRE_ENCODINGS)
                    generated_dataset = Dataset.from_pandas(generated_df)

                    sub_dataset = sample_from_dataset(generated_dataset, genre, num_examples)
                    combined_dataset = concatenate_datasets([combined_dataset, sub_dataset])
            
            print("training classifier for experiment " + e.name)
            train_classifier(combined_dataset, e.name)
            print("successfully trained classifier for experiment " + e.name)


    
