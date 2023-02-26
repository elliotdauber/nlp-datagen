from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import ast
import np
import torch
import pandas as pd

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

GENRE_ENCODINGS = {
    "gospel": 0,
    "country": 1,
    "rap": 2,
    "metal": 3
}

def get_genre_encoding(genre):
    assert(genre in GENRE_ENCODINGS)
    return GENRE_ENCODINGS[genre]

def genre_list_to_single_genre(example):
    genres = ast.literal_eval(example['genres_list'])
    genre_str = " ".join(genres)
    if "gospel" in genre_str:
        example['genre'] = get_genre_encoding("gospel")
    elif "country" in genre_str:
        example['genre'] = get_genre_encoding("country")
    elif "rap" in genre_str:
        example['genre'] = get_genre_encoding("rap")
    elif "metal" in genre_str:
        example['genre'] = get_genre_encoding("metal")
    else:
        example['genre'] = None
    return example


def clean(example):
    return {
        'text': example['lyrics'],
        'label': example['genre']
    }

def get_vanilla_dataset():
    dataset = load_dataset("brunokreiner/genius-lyrics", split='train')
    dataset = dataset.filter(lambda example: example['genres_list'] is not None and example["is_english"] == True)
    dataset = dataset.map(genre_list_to_single_genre)
    dataset = dataset.remove_columns(["Unnamed: 0", "id", "is_english", "popularity", "release_date", "artist_id", "artist_name", "artist_popularity", "artist_followers", "artist_picture_url", "genres_list"])
    dataset = dataset.filter(lambda example: example['genre'] is not None)
    return dataset

def sample_from_dataset(dataset, n_per_genre):
    # sample n_per_genre rows per genre
    sampled_data = dataset.to_pandas().groupby('genre').head(n_per_genre)
    return Dataset.from_pandas(sampled_data)

def segment_dataset(dataset):
    train_end = int(len(dataset) * 0.9)
    return DatasetDict(
        train=dataset.shuffle(seed=1111).select(range(train_end)).map(clean),
        val=dataset.shuffle(seed=1111).select(range(train_end + 1, len(dataset))).map(clean)
    )

def train_classifier(dataset):
    print("In total, there are", len(dataset), "examples")
    dataset = sample_from_dataset(dataset, 1000)
    print("After sampling, there are", len(dataset), "examples")
    split_dataset = segment_dataset(dataset)
    print(split_dataset)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=len(GENRE_ENCODINGS)).to(device)

    # Prepare the dataset - this tokenizes the dataset in batches of 64 examples.
    split_tokenized_dataset = split_dataset.map(
        lambda example: tokenizer(example['text'], padding=True, truncation=True),
        batched=True,
        batch_size=64
    )

    arguments = TrainingArguments(
        output_dir="trainer_output",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224
    )


    def compute_metrics(eval_pred):
        """Called at the end of validation. Gives accuracy"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # calculates the accuracy
        return {"accuracy": np.mean(predictions == labels)}


    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=split_tokenized_dataset['train'],
        eval_dataset=split_tokenized_dataset['val'], # change to test when you do your final evaluation!
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.predict(split_tokenized_dataset['val'])
    print(results)