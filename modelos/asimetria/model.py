# model.py — Red neuronal densa con Keras
import tensorflow as tf

def build_model(input_dim=8):
    inputs = tf.keras.Input(shape=(input_dim,), name="asimetria_features")

    x = tf.keras.layers.Dense(
        16, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_1")(x)

    x = tf.keras.layers.Dense(
        8, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_2")(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)

    return tf.keras.Model(inputs, output, name="mlp_asimetria")