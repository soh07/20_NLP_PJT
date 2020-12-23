# -*- coding: utf-8 -*-

## Setup
!pip install transformers

## Imports
import tensorflow as tf
from transformers import BertTokenizer

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import random
import time
import datetime
import os
import matplotlib.pyplot as plt


## Load Data
def read_txt(data):
    sentences, labels = [], []
    data.readline()

    while True:
        row = data.readline().split("\t")
        if row[0] != '':
            sentence, label = row[1], row[2][0]
            sentences.append(sentence)
            labels.append(label)
        else:
            break
    return sentences, labels

train = open(os.path.join('nsmc', 'ratings_train.txt'), 'r', encoding='UTF-8')
test = open(os.path.join('nsmc', 'ratings_test.txt'), 'r', encoding='UTF-8')

train_sentences, train_labels = read_txt(train)
test_sentences, test_labels = read_txt(test)

## Split Data
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_sentences, train_labels,
                                                                            test_size=0.2, random_state=42)

train_sentences = ["[CLS] " + str(sent) + " [SEP]" for sent in train_sentences]
val_sentences = ["[CLS] " + str(sent) + " [SEP]" for sent in val_sentences]
test_sentences = ["[CLS] " + str(sent) + " [SEP]" for sent in test_sentences]

## Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

train_texts = [tokenizer.tokenize(sent) for sent in train_sentences]
val_texts = [tokenizer.tokenize(sent) for sent in val_sentences]
test_texts = [tokenizer.tokenize(sent) for sent in test_sentences]

print(train_sentences[0])
print(val_sentences[0])
print(test_sentences[0])
print(train_texts[0])
print(val_texts[0])
print(test_texts[0])


## Encoding
# Max Sequence length of input tokens
max_len = 128

# convert tokens to number index
train_sequences = [tokenizer.convert_tokens_to_ids(x) for x in train_texts]
val_sequences = [tokenizer.convert_tokens_to_ids(x) for x in val_texts]
test_sequences = [tokenizer.convert_tokens_to_ids(x) for x in test_texts]

# Pad sequences to the same length (max length and 0 padding)
train_sentences = np.array(
    pad_sequences(train_sequences, maxlen=max_len, dtype="long", truncating="post", padding="post"))
val_sentences = np.array(
    pad_sequences(val_sequences, maxlen=max_len, dtype="long", truncating="post", padding="post"))
test_sentences = np.array(
    pad_sequences(test_sequences, maxlen=max_len, dtype="long", truncating="post", padding="post"))

train_labels = np.array(train_labels, dtype='float')
val_labels = np.array(val_labels, dtype='float')
test_labels = np.array(test_labels, dtype='float')


## Building model
inputs = tf.keras.layers.Input(shape=[max_len])
embedding = tf.keras.layers.Embedding(tokenizer.vocab_size + 1, 64, input_length=max_len)(inputs)
lstm = tf.keras.layers.LSTM(128)(embedding)
dense = tf.keras.layers.Dense(256, activation='relu')(lstm)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


## Check the weight of model
weights = model.weights

for weight in weights:
    print(weight.name, weight.shape)


## Training Model
history = model.fit(train_sentences, 
                    train_labels, 
                    validation_data=(val_sentences, val_labels),
                    batch_size=32,
                    epochs=50)

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(loss)+1)

## Visualize the model fitting
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_fitting_model.png')


## Calculate the performance of model
test_pred = model.predict(test_sentences)
print("Output Shape: {}\n".format(test_pred.shape))
print("Prediction Shape:", test_pred[0].shape)
print("Sum:", np.sum(test_pred[0]))
print("Argmax:", np.argmax(test_pred[0]))

print(np.argmax(test_pred, axis=1))
print(test_labels)

result = model.evaluate(test_sentences, test_labels)
print("Results:", result)