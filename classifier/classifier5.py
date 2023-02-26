from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import ast
import np


# DataLoader(zip(list1, list2))

def get_first_genre(example):
    genres = ast.literal_eval(example['genres_list'])
    genre_str = " ".join(genres)
    if "gospel" in genre_str:
        example['genre'] = 0
    elif "country" in genre_str:
        example['genre'] = 1
    elif "rap" in genre_str:
        example['genre'] = 2
    elif "metal" in genre_str:
        example['genre'] = 3
    else:
        example['genre'] = None
    return example


# dataset = load_dataset("imdb", split="train")

# # Just take the first 50 tokens for speed/running on cpu
# def truncate(example):
#     return {
#         'text': " ".join(example['text'].split()[:50]),
#         'label': example['label']
#     }

def truncate(example):
    return {
        'text': " ".join(example['lyrics'].split()[:50]),
        'label': example['genre']
    }

dataset = load_dataset("brunokreiner/genius-lyrics", split='train')
print(len(dataset))
dataset = dataset.filter(lambda example: example['genres_list'] is not None and example["is_english"] == True)
dataset = dataset.map(get_first_genre)
dataset = dataset.remove_columns(["Unnamed: 0", "id", "is_english", "popularity", "release_date", "artist_id", "artist_name", "artist_popularity", "artist_followers", "artist_picture_url", "genres_list"])
dataset = dataset.filter(lambda example: example['genre'] is not None)
print("There are", len(dataset), "examples")

print(dataset.select(range(5)))
for i in range(5):
    print(dataset[i])
    print()
# exit(0)

# Take 128 random examples for train and 32 validation
small_dataset = DatasetDict(
    train=dataset.shuffle(seed=1111).select(range(128)).map(truncate),
    val=dataset.shuffle(seed=1111).select(range(128, 160)).map(truncate),
)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

# Prepare the dataset - this tokenizes the dataset in batches of 16 examples.
small_tokenized_dataset = small_dataset.map(
    lambda example: tokenizer(example['text'], padding=True, truncation=True),
    batched=True,
    batch_size=16
)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=4)

arguments = TrainingArguments(
    output_dir="trainer_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.predict(small_tokenized_dataset['val'])
print(results)