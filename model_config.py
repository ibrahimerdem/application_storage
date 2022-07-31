import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    input_item = layers.Input(shape=(hparams["n_padded"],),
                              name="item_ids",
                              dtype=tf.int32)
    input_price = layers.Input(shape=(hparams["n_padded"], hparams["price_max"]),
                               name="price",
                               dtype=tf.float32)
    input_category = layers.Input(shape=(hparams["n_padded"], hparams["category_max"]),
                                  name="category",
                                  dtype=tf.float32)
    input_recency = layers.Input(shape=(hparams["n_padded"], hparams["recency_max"]),
                                 name="recency",
                                 dtype=tf.float32)
    input_month = layers.Input(shape=(hparams["n_padded"], hparams["month_max"]),
                               name="month",
                               dtype=tf.float32)
    input_dayofweek = layers.Input(shape=(hparams["n_padded"], hparams["dayofweek_max"]),
                                   name="dayofweek",
                                   dtype=tf.float32)
    input_payment = layers.Input(shape=(hparams["n_padded"], hparams["payment_max"]),
                                 name="payment",
                                 dtype=tf.float32)
    input_customer_since = layers.Input(shape=(hparams["since_max"],),
                                        name="c_since",
                                        dtype=tf.float32)
    encoding_padding_mask = tf.math.logical_not(tf.math.equal(input_item, 0))
    embedding_item = layers.Embedding(input_dim=hparams["item_max"]+1,
                                      output_dim=hparams["emb_unit"])(input_item)
 
    if hparams["model"]==1:
        input_list = [input_item]
        concat_inputs = embedding_item
    elif hparams["model"]==2:
        input_list = [input_item,
                      input_recency,
                      input_month,
                      input_dayofweek,
                      input_payment]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              input_recency,
                                              input_month,
                                              input_dayofweek,
                                              input_payment])
        concat_inputs = layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==3:
        input_list = [input_item,
                      input_recency,
                      input_month,
                      input_dayofweek,
                      input_payment,
                      input_price,
                      input_category]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              input_recency,
                                              input_month,
                                              input_dayofweek,
                                              input_payment,
                                              input_price,
                                              input_category])
        concat_inputs = layers.BatchNormalization()(concat_inputs)
    elif hparams["model"]==4:
        input_list = [input_item,
                      input_recency,
                      input_month,
                      input_dayofweek,
                      input_payment,
                      input_price,
                      input_category,
                      input_customer_since]
        concat_inputs = layers.Concatenate()([embedding_item,
                                              input_recency,
                                              input_month,
                                              input_dayofweek,
                                              input_payment,
                                              input_price,
                                              input_category])
        concat_inputs = tf.keras.layers.BatchNormalization()(concat_inputs)
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
    if hparams["model"]==4:
        r = layers.RepeatVector(hparams["n_padded"])(input_list[-1])
        x = layers.Concatenate()([x, r])
    z = layers.Attention(use_scale=False,
                         causal=True)(inputs=[x, x],
                                      mask=[encoding_padding_mask,
                                            encoding_padding_mask])
    output = layers.Dense(hparams["item_max"]+1)(z)
    model = keras.Model(input_list, output)
    model.compile(loss=loss_function,
                  optimizer=keras.optimizers.Adam(learning_rate=hparams["learning"]),
                  metrics=["sparse_categorical_accuracy"])
    return model
