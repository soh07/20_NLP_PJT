# -*- coding: utf-8 -*-

## Setup
!pip install bert-tensorflow
!pip install -q tf-models-official==2.3.0
!pip install transformers


## Imports
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import ElectraModel, ElectraTokenizer, TFElectraModel

import numpy as np
import json
import os

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks



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

train= open(os.path.join("/ratings_train.txt"),'r', encoding='UTF-8')
test= open(os.path.join("/ratings_test.txt"),'r', encoding='UTF-8')

train_sentences, train_labels = read_txt(train)
test_sentences, test_labels = read_txt(test)

print(train_sentences[0:5])
print(train_labels[0:5])


## Split Data
from sklearn.model_selection import train_test_split
train_sentences, dev_sentences, train_labels, dev_labels = train_test_split(train_sentences, train_labels,
                                                                            test_size=0.2, random_state=42)

##
oov_token = "<OOV>"
pad_type = 'post'
truc_type = 'post'
embedding_size = 64
max_len = 128   
#output_size = 2


## Preprocess the data
# Tokenizing & Encoding
text_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)
   
    
def bert_encode(sentences, tokenizer):
  num_examples = len(sentences)
  
  sentence = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in np.array(sentences)])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence.shape[0]
  input_word_ids = tf.concat([cls, sentence], axis=-1).to_tensor()

  input_mask = tf.ones_like(input_word_ids)
  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence)
  input_type_ids = tf.concat(
      [type_cls, type_s1], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids,
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs


train = bert_encode(train_sentences, text_tokenizer)
train_labels = np.expand_dims(train_labels, axis=-1).astype(float)

dev = bert_encode(dev_sentences, text_tokenizer)
dev_labels = np.expand_dims(dev_labels, axis=-1).astype(float)

test = bert_encode(test_sentences, text_tokenizer)
test_labels = np.expand_dims(test_labels, axis=-1).astype(float)


print('---Training Data---')
for key, value in train.items():
      print(f'{key:15s} shape: {value.shape}')
print('Training labels', train_labels.shape)

print('---Validation Data---')
for key, value in dev.items():
      print(f'{key:15s} shape: {value.shape}')
print('Validation labels', dev_labels.shape)

print('---Test Data---')
for key, value in test.items():
      print(f'{key:15s} shape: {value.shape}')
print('Test labels', test_labels.shape)


## Restore the encoder weights
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

print(config_dict)

bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=2)

checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()


## Build the model
input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                   name="input_mask")
input_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                                    name="input_type_ids")
bert_layer = TFElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", from_pt=True)
outputs = bert_layer([input_word_ids, input_mask, input_type_ids])
net = tf.keras.layers.Dense(128, activation='relu')(outputs.last_hidden_state)
net = tf.keras.layers.Dense(256, activation='relu')(net)
net = tf.keras.layers.Dense(1, activation="sigmoid", name='classifier')(net)

model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[net])
model.summary()


## Set up the optimizer
# Set up epochs and steps
epochs = 4
batch_size = 32
eval_batch_size = 32

train_data_size = len(train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(2e-5,
                                              num_train_steps=num_train_steps, 
                                              num_warmup_steps=warmup_steps)
print(type(optimizer))
print(f"train data size : {train_data_size}")
print(f"steps per epochs : {steps_per_epoch}")
print(f"num train steps : {num_train_steps}")
print(f"warmup steps : {warmup_steps}")


## Train the model
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy',dtype=None)]
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(train, train_labels,
          validation_data=(dev, dev_labels),
          batch_size=batch_size,
          epochs=4,)


## Calculate the performance of model
test_pred = model.predict(test)
print("Output Shape: {}\n".format(test_pred.shape))
print(test_pred)
print(np.argmax(test_pred, axis=1))
print(model.evaluate(test, test_labels))