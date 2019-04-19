import pandas as pd
import numpy as np
import pickle
import os
from lxml import etree
from logos_tools import *
from logos_opta import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Conv2D, Flatten, MaxPool2D, Reshape, Concatenate, Dropout, Add, AveragePooling2D, Bidirectional, GRU
from tensorflow.keras.models import Model
from absl import flags, logging, app

FLAGS = flags.FLAGS
flags.DEFINE_string('sample_dir', None, 'sample_dir')
flags.DEFINE_string('model_dir', None, 'model_dir')
flags.DEFINE_string('dim_game', None, 'dim_game')
flags.DEFINE_string('tokenizer_file', None, 'tokenizer_file')
flags.DEFINE_string('model_json', None, 'model_json')
flags.DEFINE_string('weights_file', None, 'weights_file')
flags.DEFINE_integer('patience', None, 'early stopping patience')


def main(_):
    sample_dir = FLAGS.sample_dir
    model_dir = FLAGS.model_dir
    dim_game = FLAGS.dim_game
    patience = FLAGS.patience
    tokenizer_file = FLAGS.tokenizer_file
    model_json = FLAGS.model_json
    weights_file = FLAGS.weights_file
    test_rounds = [19]
    sub_sample_step = 2

    # get sample files
    samples = [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f)) and '.DS' not in f]

    # read
    data_list = []
    for x in samples:
        data = pd.read_csv(os.path.join(sample_dir, x), sep='\t')
        data['game'] = x.split('.')[0]
        data_list.append(data)
    data = pd.concat(data_list, axis=0, sort=False)

    # sub-sample
    data.index = data.game
    data = data.groupby(data.index).apply(sub_sample, step=sub_sample_step)
    data = data.reset_index(drop=True)

    # 添加新特征
    data = add_new_features(data)

    # select useful fields
    self_fields = {'pass1', 'front1', 'defend', 'ballTouch', 'shoot', 'foul', 'clearance', 'takeon1',
                'positionC1', 'positionL1', 'positionR1', 'positionB1'}
    opp_fields = self_fields.copy()
    useful_fields = {'period', 'last_team', 'last_position', 'last1_event', 'last2_event', 'last_x', 'last_y', 'last10_list', 'game'}
    label_fields = {'y_team', 'y_x', 'y_y'}

    # merge home and away team features
    data.index = data.game+','+data.start.astype('str')+','+data.end.astype('str')
    home_df = data.loc[data.team_id==1, self_fields.union(useful_fields).union(label_fields)]
    away_df = data.loc[data.team_id==0, opp_fields]
    away_df.columns = away_df.columns + '_o'
    opp_fields = set(away_df.columns)
    data = pd.concat([home_df, away_df], axis=1, sort=False)
    data = data.reset_index(drop=True)
    
    # transform features
    num_features = self_fields.union(opp_fields)
    str_features = {'period', 'last_team', 'last_position', 'last1_event', 'last2_event'}
    column_trans = ColumnTransformer(
        [('onehot', OneHotEncoder(handle_unknown='ignore'), list(str_features)),
        ('standard', StandardScaler(), list(num_features))], 
        remainder='drop')
    X = column_trans.fit_transform(data)
    # with open(os.path.join(model_dir, transformer_file), 'wb') as handle:
    #     pickle.dump(column_trans, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # select specified rounds as valid set
    train_index, test_index = get_train_test_index_from_rounds(dim_game, data.game, test_rounds)
    y = data.y_team
    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]
    
    # lstm
    docs_seg = [x.split(',') for x in data.last10_list]
    t = keras.preprocessing.text.Tokenizer(oov_token=1)
    t.fit_on_texts(docs_seg)
    with open(os.path.join(model_dir, tokenizer_file), 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # sequence to word index
    encoded_docs = t.texts_to_sequences(docs_seg)
    vocabulary_size = len(t.word_index)+1
    # padding
    sequence_length = 10
    padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=sequence_length, padding='pre')
    # train test set
    seq_train, seq_valid = padded_docs[train_index], padded_docs[test_index]
    # buld up model
    embedding_dim = 300
    drop_rate = 0.5
    l2_constraint = 3
    rnn_units = 100
    main_input = Input(shape=(sequence_length,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, 
                        input_length=sequence_length, trainable=True, name='embedding')(main_input)
    rnn = GRU(units=rnn_units, return_sequences=False, 
            kernel_constraint=keras.constraints.MaxNorm(max_value=l2_constraint), 
            name='rnn')(embedding)
    seq_output = Dense(units=1, activation='sigmoid', name='seq_output')(rnn)
    model = Model(inputs=main_input, outputs=seq_output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    with open(os.path.join(model_dir, model_json), 'w') as f:
        f.write(model.to_json())

    # save while training
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            # os.path.join(model_dir, 'next_team_rnn.{epoch:02d}-{val_acc:.2f}.hdf5'), 
            os.path.join(model_dir, weights_file), 
            monitor='val_acc',
            save_weights_only=True, 
            save_best_only=True)
    
    # early stopping
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_acc', patience=patience)

    # training
    batch_size = 32
    epochs = 100
    result = model.fit(x=seq_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, 
                    validation_data=(seq_valid, y_valid), callbacks = [checkpoint_callback, earlystopping_callback])

if __name__ == '__main__':
    app.run(main)