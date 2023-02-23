import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Load the labeled song lyric CSV file
data = pd.read_csv('data/clean-data.csv')

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
labels = torch.tensor(data.Genre0.values)

# Split the data into training and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                    random_state=2022,
                                                                                    test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                       random_state=2022, test_size=0.1)

# Define the fine-tuning model architecture
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False,
)

# Define the optimizer and the learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4

# Fine-tune the model on the training set
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in range(len(train_inputs)):
        optimizer.zero_grad()
        outputs = model(train_inputs[batch], token_type_ids=None, attention_mask=train_masks[batch],
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
