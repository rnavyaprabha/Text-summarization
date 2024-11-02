##
# CS 59000-09 Group Project
# 
##

# refered few sites to comeup with final code here
#  https://analyticsindiamag.com/sequence-to-sequence-modeling-using-lstm-for-language-translation/ 
# and 
# https://www.kaggle.com/code/sudiptodip/seq2seq-enc-dec

#import required Libraries
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate, TimeDistributed
from keras.callbacks import EarlyStopping
from rouge import Rouge

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Tokenize and preprocess article and highlights
t_maxlen = 100 # maximum length of article
s_maxlen = 20 # maximum length of highlights
t_tokenizer = Tokenizer()
t_tokenizer.fit_on_texts(train_df['article'])
s_tokenizer = Tokenizer()
s_tokenizer.fit_on_texts(train_df['highlights'])
t_vocab_size = len(t_tokenizer.word_index) + 1
s_vocab_size = len(s_tokenizer.word_index) + 1
#train_x and test_x are the tokenized and padded article sequences, while train_y and test_y are the tokenized and padded highlight sequences
train_x = pad_sequences(t_tokenizer.texts_to_sequences(train_df['article']), maxlen=t_maxlen, padding='post')
train_y = pad_sequences(s_tokenizer.texts_to_sequences(train_df['highlights']), maxlen=s_maxlen, padding='post')
test_x = pad_sequences(t_tokenizer.texts_to_sequences(test_df['article']), maxlen=t_maxlen, padding='post')
test_y = pad_sequences(s_tokenizer.texts_to_sequences(test_df['highlights']), maxlen=s_maxlen, padding='post')
#Add special tokens (<start>, <end>, <pad>, and <unk>) to the s_tokenizer word index
#will be used to indicate the beginning and end of each highlight sequence during training and testing
s_tokenizer.word_index['<start>'] = s_vocab_size
s_tokenizer.word_index['<end>'] = s_vocab_size + 1
#will be used for padding and out-of-vocabulary words
s_tokenizer.word_index['<pad>'] = 0
s_tokenizer.word_index['<unk>'] = s_vocab_size + 2
#Converting the text data in the 'highlights' column of the 'train_df' DataFrame into tokenized sequences using a tokenizer
#Adding special start and end tokens to each sequence
train_y_in = s_tokenizer.texts_to_sequences(train_df['highlights'].apply(lambda x: '<start> ' + x))
train_y_out = s_tokenizer.texts_to_sequences(train_df['highlights'].apply(lambda x: x + ' <end>'))
train_y_in = pad_sequences(train_y_in, maxlen=s_maxlen, padding='post')
train_y_out = pad_sequences(train_y_out, maxlen=s_maxlen, padding='post')

# Define the model architecture
latent_dim = 128
# Encoder part
enc_input = Input(shape=(t_maxlen, ))
enc_embed = Embedding(t_vocab_size, latent_dim, input_length=t_maxlen)(enc_input)
enc_lstm = Bidirectional(LSTM(latent_dim, return_state=True))
enc_outputs, enc_fh, enc_fc, enc_bh, enc_bc = enc_lstm(enc_embed)
enc_h = Concatenate(axis=-1, name='enc_h')([enc_fh, enc_bh])
enc_c = Concatenate(axis=-1, name='enc_c')([enc_fc, enc_bc])
# Decoder part
dec_input = Input(shape=(None, ))
dec_embed = Embedding(s_vocab_size, latent_dim)(dec_input)
dec_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
dec_outputs, _, _ = dec_lstm(dec_embed, initial_state=[enc_h, enc_c])
dec_dense = TimeDistributed(Dense(s_vocab_size, activation='softmax'))
dec_output = dec_dense(dec_outputs)
#Defines the model architecture with two inputs (encoder input and decoder input) and one output (decoder output).
model = Model([enc_input, dec_input], dec_output)
#Compiles the model with a loss function of sparse categorical cross-entropy and an optimizer of RMSprop
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
#trains a sequence-to-sequence model on the training data
# validates it on a subset of the data using early stopping with a maximum of 50 epochs and a patience of 5, while monitoring the validation loss
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

history = model.fit(
    [train_x, train_y_in], np.expand_dims(train_y_out, -1),
    batch_size=128,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop]
)


#Evaluate the model on test data
test_loss = model.evaluate([test_x, test_y[:,:-1]], test_y[:,1:], verbose=0)
print("Test Loss:", test_loss)

#Generate predictions on test data
preds = model.predict([test_x, test_y[:,:-1]])

#Convert predictions into text (summary) format
inv_s_tokenizer = {v:k for k, v in s_tokenizer.word_index.items()}
preds_text = []
for p in preds:
    summary = ''
    for val in p:
        if val > 0:
            summary += inv_s_tokenizer[val] + ' '
    preds_text.append(summary[:-1])

# Convert predictions into text (summary) format
inv_s_tokenizer = {v:k for k, v in s_tokenizer.word_index.items()}
preds_text = []
for p in preds:
    summary = ''
    for val in p:
        if val > 0:
            summary += inv_s_tokenizer[val] + ' '
    preds_text.append(summary[:-1])

# Create a new DataFrame with the predicted summaries and the corresponding texts
output_df = pd.DataFrame({'Article': test_df['article'], 'Actual Summary': test_df['highlights'], 'Predicted Summary': preds_text})

# Write the DataFrame to a CSV file
output_df.to_csv('output.csv', index=False)


# Load the output DataFrame
output_df = pd.read_csv('output.csv')

# Initialize the ROUGE scorer
rouge = Rouge()

# Calculate ROUGE scores for each predicted summary
rouge_scores = []
for i, row in output_df.iterrows():
    predicted_summary = row['Predicted Summary']
    actual_summary = row['Actual Summary']
    scores = rouge.get_scores(predicted_summary, actual_summary)[0]
    rouge_scores.append(scores)

# Add the ROUGE scores to the output DataFrame
rouge_df = pd.DataFrame(rouge_scores)
output_df = pd.concat([output_df, rouge_df], axis=1)

# Print the average ROUGE scores
print("ROUGE-1: ", np.mean(output_df['rouge-1']['f']))
print("ROUGE-2: ", np.mean(output_df['rouge-2']['f']))
print("ROUGE-L: ", np.mean(output_df['rouge-l']['f']))
print("ROUGE-S: ", np.mean(output_df['rouge-s']['f']))


