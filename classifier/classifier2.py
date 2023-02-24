import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Set the number of labels and the device to be used
NUM_LABELS = 4
BATCH_SIZE = 2
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS).to(device)

# Load the labeled song lyric CSV file
data = pd.read_csv('data/clean-data.csv')
data['Genre'] = data['Genre'].astype('category')
data = data.groupby('Genre').head(128)
print("encoding data...")
# Tokenize the lyrics using the BERT tokenizer
encoded_data = tokenizer.batch_encode_plus(
    data.Lyric.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)
print("encoded data...")

# Convert the tokenized lyrics to a BERT-compatible format
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(data.Genre.cat.codes.values)

# Split the data into training and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.25)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42, test_size=0.25)
print("split the dataset...")
print(train_inputs.shape)

# Create a DataLoader for each dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_dataset)
validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=BATCH_SIZE)
print("dataloader done...")

# Set up the optimizer and training loop
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4

for epoch in range(epochs):
    print("in epoch", epoch)
    # Train the model for one epoch
    model.train()
    iteration = 0
    print(train_dataloader)
    for batch in train_dataloader:
        print("in batch", iteration)
        iteration+= 1
        # Add batch to device
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = torch.unsqueeze(b_labels, dim=1)
        b_labels = torch.broadcast_to(b_labels, (BATCH_SIZE, NUM_LABELS))
        # Forward pass
        print("running forward...")
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels.float())
        print("running backward...")
        loss = outputs.loss
        # Backward pass
        loss.backward()
        # Update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        
    # Evaluate the model after each epoch
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    print("evald")

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = torch.unsqueeze(b_labels, dim=1)
        b_labels = torch.broadcast_to(b_labels, (BATCH_SIZE, NUM_LABELS))
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels.float())
            loss = outputs.loss
            logits = outputs.logits
        # Calculate the accuracy for this batch of validation sentences
        pred_labels = torch.argmax(logits, axis=1)
        s = torch.sum(pred_labels == b_labels.argmax(dim=1)).item()
        # print(s.shape)
        # print(b_labels.shape)
        tmp_eval_accuracy = s / len(b_labels)
        eval_loss += loss.item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    # Report the final accuracy and loss values
    print("Validation Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("Validation Loss: {0:.2f}".format(eval_loss/nb_eval_steps))

model.save_pretrained('model')