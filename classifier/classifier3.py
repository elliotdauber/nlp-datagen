from datasets import load_dataset, DatasetDict, Dataset, Features
from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import np
import pandas as pd
from sklearn.model_selection import train_test_split


# Load the DataFrame
df = pd.read_csv('data/clean-data.csv')
df = df.groupby('Genre').head(10)

# Define the function to convert a row to a dictionary
dataset = Dataset.from_pandas(df.rename(columns={'Lyric': 'text', 'Genre': 'label'}))
print(dataset[0])

print(dataset)
# Tokenize the datasets
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

for example in dataset:
    example['text'] = tokenizer(example['text'], padding=True, truncation=True, max_length=5000)

# Split the dataset into training and validation sets (80/20 split)
train_size = int(len(dataset)*0.8)
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))

# train_dataset = train_dataset.map(lambda example: tokenizer(example['text'], padding=True, truncation=True, max_length=5000), batched=True)
# val_dataset = val_dataset.map(lambda example: tokenizer(example['text'], padding=True, truncation=True, max_length=5000), batched=True)

# train_dataset.map(lambda example: print(len(example['text'])))

# Rename the label column to 'labels'
train_dataset = train_dataset.rename_column('label', 'labels')
val_dataset = val_dataset.rename_column('label', 'labels')




model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=4)

arguments = TrainingArguments(
    output_dir="trainer_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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
    train_dataset=train_dataset,
    eval_dataset=val_dataset, # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.predict(val_dataset)
print(results)