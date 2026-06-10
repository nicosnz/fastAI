import tensorflow as tf

def build_model(input_dim=8):
    inputs = tf.keras.Input(shape=(input_dim,), name="asimetria_features")

    # Capa 1: 16 neuronas
    # Más neuronas que inputs (8) para que la red tenga espacio
    # de aprender combinaciones de features
    x = tf.keras.layers.Dense(
        16, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_1")(x)

    # Capa 2: 8 neuronas
    # Se reduce a la mitad — fuerza compresión de la información
    # El modelo aprende solo lo más relevante
    x = tf.keras.layers.Dense(
        8, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_2")(x)

    # Output: 1 neurona con sigmoid → probabilidad entre 0 y 1
    # > 0.5 = ACV, ≤ 0.5 = Normal
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)

    return tf.keras.Model(inputs, output, name="mlp_asimetria")