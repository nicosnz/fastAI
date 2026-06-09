import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)

# cargar modelo entrenado
model = tf.keras.models.load_model("modelo_final_asimetria.keras")

# ruta de la imagen nueva
img_path = "test/test2.jpg"

# cargar imagen
img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)

# convertir a array
img_array = tf.keras.utils.img_to_array(img)

# normalizar
img_array = img_array / 255.0

# agregar batch dimension
img_array = np.expand_dims(img_array, axis=0)

# predicción
pred = model.predict(img_array)[0][0]

print("Probabilidad:", pred)

# interpretación
if pred > 0.5:
    print("➡️ NORMAL")
else:
    print("➡️ ACV")