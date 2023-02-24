import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

NUM_LABELS = 4

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
print("after tokenizer")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
print("after model")



# Load the labeled song lyric CSV file
data = pd.read_csv('data/clean-data.csv')
data['Genre'] = data['Genre'].astype('category')
print(data['Lyric'].dtype)

data = data.groupby('Genre').head(128)
print(data.shape[0])


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

# Convert the tokenized lyrics to a BERT-compatible format
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(data.Genre.cat.codes.values)

batch_size, seq_length = input_ids.size()
print(input_ids.shape)

divisor = 512
num_batches = batch_size // divisor
print(num_batches)
input_ids = input_ids.narrow(0, 0, num_batches * divisor).view(-1, divisor)
attention_masks = attention_masks.narrow(0, 0, num_batches * divisor).view(-1, divisor)
print(input_ids.shape)
print(attention_masks.shape)


# Split the data into training and validation sets
num_samples = min(len(input_ids), len(labels))
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids[:num_samples], labels[:num_samples],
                                                                                    random_state=2022,
                                                                                    test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks[:num_samples], input_ids[:num_samples],
                                                       random_state=2022, test_size=0.1)
# Define the fine-tuning model architecture
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    output_attentions=False,
    output_hidden_states=False,
)

# Define the optimizer and the learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4

# Fine-tune the model on the training set
for epoch in range(epochs):
    print("in epoch", epoch)
    model.train()
    train_loss = 0
    print(len(train_inputs))
    for batch in range(len(train_inputs)):
        optimizer.zero_grad()
        print(batch)
        print(train_inputs.shape)
        print(train_masks.shape)
        print(train_labels.shape)
        outputs = model(train_inputs[batch].resize(1, 512), token_type_ids=None, attention_mask=train_masks[batch].resize(1, 512),
                        labels=train_labels[batch])
        loss = outputs[0]
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in range(len(validation_inputs)):
        with torch.no_grad():
            outputs = model(validation_inputs[batch], token_type_ids=None,
                            attention_mask=validation_masks[batch])
            logits = outputs[0]
        label_ids = validation_labels[batch]
        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Epoch {} - Training loss: {}".format(epoch + 1, train_loss / len(train_inputs)))
    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))

# Save the trained model
model.save_pretrained('data/output.csv')
