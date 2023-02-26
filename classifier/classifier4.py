from datasets import load_dataset, DatasetDict, Dataset, Features
from transformers import AutoTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import np
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import random

# Load the DataFrame
df = pd.read_csv('data/clean-data.csv')
df = df.head(8)
def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# apply the function to the 'Lyric' column
df['Lyric'] = df['Lyric'].apply(lambda x: random_string(2))
print(df)

# Define the function to convert a row to a dictionary
dataset = Dataset.from_pandas(df.rename(columns={'Lyric': 'text', 'Genre': 'label'}))

# Tokenize the datasets
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

# INSERT CODE HERE

# Tokenize the text data
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# Apply tokenization to the dataset
# dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
dataset = dataset.map(tokenize, batched=False)

# Split the dataset into train and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

# END INSERT CODE


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