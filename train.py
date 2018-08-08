# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse
import numpy as np
import pandas as pd

import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras_pickle_wrapper import KerasPickleWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pickle as pickle

from azureml.logging import get_azureml_logger

EMBEDDING_FILE = 'C:/Keshav/DataScience/ToxicCommentClassification/all/glove840b300dtxt/glove.840B.300d.txt'
train = pd.read_csv('C:/Keshav/DataScience/ToxicCommentClassification/all/train.csv')

train["comment_text"].fillna("fillna")
X_train = train["comment_text"].str.lower()
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

max_features=100000
maxlen=150
embed_size=300

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            
tok=text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(list(X_train))
X_train=tok.texts_to_sequences(X_train)
x_train=sequence.pad_sequences(X_train,maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
        
sequence_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
preds = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])

batch_size = 128
epochs = 1
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)

filepath="outputs/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
callbacks_list = [ra_val,checkpoint, early]

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)

# keras models can not be pickled, use KerasPickleWrapper to wrap the model for serialization
model.save('outputs/Toxic_Trained_model.h5')
LoadModel = load_model('outputs/Toxic_Trained_model.h5')
model_wrapper = KerasPickleWrapper(LoadModel)
f = open('outputs/Toxic_Trained_model.pkl', 'wb')
pickle.dump(model_wrapper,f)
f.close()

f2 = open('outputs/Toxic_Trained_model.pkl', 'rb')
clf2 = pickle.load(f2)


Test_Text = " What the crap, fuck mate. Dont buy this product, I hate it."
Test_Text=[Test_Text]
Test_Text_df=pd.DataFrame(Test_Text)
Test_Text_df.columns = ['comment_text']

Test_Text_Val = Test_Text_df["comment_text"].str.lower()
Test_Text_Val=tok.texts_to_sequences(Test_Text_Val)
Test_Text_Val=sequence.pad_sequences(Test_Text_Val,maxlen=maxlen)
#print(Test_Text_Val)
Test_Text_Pred = clf2().predict(Test_Text_Val,verbose=1)
#print(Test_Text_Pred)
columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
Test_Text_Pred_df = pd.DataFrame(np.atleast_2d(Test_Text_Pred), columns=columns)
print(Test_Text_Pred_df)

# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()
print(args)

# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)