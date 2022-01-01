import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#@tf.autograph.experimental.do_not_convert
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object_ = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, 
      reduction="none"
      ) 
    loss_ = loss_object_(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def inputs(hparams):
    input_item = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                              dtype=tf.int32)
    input_rec = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                             dtype=tf.int32)
    input_mon = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                             dtype=tf.int32)
    input_day = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                             dtype=tf.int32)
    input_pay = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                             dtype=tf.int32)
    input_met = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                             dtype=tf.int32)
    input_cat = layers.Input(batch_input_shape=[None, hparams["n_padded"]],
                             dtype=tf.int32)
    encoding_padding_mask = tf.math.logical_not(tf.math.equal(input_item, 0))
    embedding_item = layers.Embedding(input_dim=hparams["item_max"],
                                      output_dim=hparams["emb_unit"])(input_item)
    embedding_rec = layers.Embedding(input_dim=hparams["rec_max"],
                                     output_dim=hparams["emb_unit"])(input_rec)
    embedding_mon = layers.Embedding(input_dim=hparams["mon_max"],
                                     output_dim=hparams["emb_unit"])(input_mon)
    embedding_day = layers.Embedding(input_dim=hparams["day_max"],
                                     output_dim=hparams["emb_unit"])(input_day) 
    embedding_pay = layers.Embedding(input_dim=hparams["pay_max"],
                                     output_dim=hparams["emb_unit"])(input_pay)
    embedding_met = layers.Embedding(input_dim=hparams["pay_max"],
                                     output_dim=hparams["emb_unit"])(input_met)
    embedding_cat = layers.Embedding(input_dim=hparams["cat_max"],
                                     output_dim=hparams["emb_unit"])(input_cat)   
    if hparams["model"]==1:
        input_list = [input_item]
        concat_inputs = embedding_item
    elif hparams["model"]==2:
        input_list = [input_item,
                      input_rec]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec])
        concat_inputs = layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==3:
        input_list = [input_item,
                      input_rec,
                      input_mon]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec,
                                              embedding_mon])
        concat_inputs = layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==4:
        input_list = [input_item,
                      input_rec,
                      input_mon,
                      input_day]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec,
                                              embedding_mon,
                                              embedding_day])
        concat_inputs = layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==5:
        input_list = [input_item,
                      input_rec,
                      input_mon,
                      input_day,
                      input_pay]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec,
                                              embedding_mon,
                                              embedding_day,
                                              embedding_pay])
        concat_inputs = tf.keras.layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==6:
        input_list = [input_item,
                      input_rec,
                      input_mon,
                      input_day,
                      input_met]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec,
                                              embedding_mon,
                                              embedding_day,
                                              embedding_met])
        concat_inputs = tf.keras.layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==7:
        input_list = [input_item,
                      input_rec,
                      input_mon,
                      input_day,
                      input_cat]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec,
                                              embedding_mon,
                                              embedding_day,
                                              embedding_cat])
        concat_inputs = layers.BatchNormalization()(concat_inputs) 
    else:
        input_list = [input_item,
                      input_rec,
                      input_mon,
                      input_day,
                      input_pay,
                      input_cat]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              embedding_rec,
                                              embedding_mon,
                                              embedding_day,
                                              embedding_pay,
                                              embedding_cat])
        concat_inputs = layers.BatchNormalization()(concat_inputs) 
    return input_list, concat_inputs, encoding_padding_mask
        
def model_base(hparams):
    input_list, concat_inputs, encoding_padding_mask = inputs(hparams)
    if hparams["style"]=="gru":
        if hparams["bidirect"]:
            x = layers.Bidirectional(layers.GRU(units=hparams["rnn_unit"],
                                                return_sequences=True))(concat_inputs)
        else:
            x = layers.GRU(units=hparams["rnn_unit"],
                           return_sequences=True)(concat_inputs)
    else:
        if hparams["bidirect"]:
            x = layers.Bidirectional(layers.LSTM(units=hparams["rnn_unit"],
                                                 return_sequences=True))(concat_inputs)
        else:
            x = layers.LSTM(units=hparams["rnn_unit"],
                            return_sequences=True)(concat_inputs)
    x = layers.Dropout(rate=hparams["dropout"])(x)
    x = layers.BatchNormalization()(x)
    att = layers.Attention(use_scale=False,
                           causal=True)(inputs=[x, x],
                                        mask=[encoding_padding_mask,
                                              encoding_padding_mask])
    output = layers.Dense(hparams["item_max"])(att)
    model = keras.Model(input_list, output)
    model.compile(loss=loss_function,
                  optimizer=keras.optimizers.Adam(learning_rate=hparams["learning"]),
                  metrics=["sparse_categorical_accuracy"])
    return model
