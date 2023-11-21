# Importa la biblioteca NumPy con el alias np para realizar operaciones matemáticas eficientes
import numpy as np

# Importa la biblioteca os para interactuar con el sistema operativo
import os

# Importa TensorFlow
import tensorflow as tf

# Imprime la versión de TensorFlow
#print(tf._version_)

# Crea un array de valores desde -10.0 hasta 10.0 con un paso de 0.01
X = np.arange(-10.0, 10.0, 1e-2)

# Mezcla aleatoriamente los valores en el array X
np.random.shuffle(X)

# Crea un array y con una relación cuadrática y = 0.5 * X^2 - 3 * X + 2
y = 0.5 * X**2 - 3 * X + 2

# Calcula los índices para dividir los datos en conjuntos de entrenamiento, validación y prueba
train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))

# Divide los datos en conjuntos de entrenamiento, validación y prueba
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

# Limpia la sesión de TensorFlow para asegurar un nuevo modelo
tf.keras.backend.clear_session()

# Crea un modelo secuencial de TensorFlow con una capa densa (fully connected) de una unidad
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

# Compila el modelo con un optimizador de descenso de gradiente estocástico (SGD) y función de pérdida de error cuadrático medio
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)

# Imprime un resumen del modelo, mostrando la arquitectura y el número de parámetros
print(linear_model.summary())

# Entrena el modelo con los datos de entrenamiento y valida con los datos de validación durante 20 épocas
linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

# Imprime las predicciones del modelo para algunos valores de entrada específicos
print(linear_model.predict([[0.0], [2.0], [3.1], [4.2], [5.2]]).tolist())

# Define la ruta para guardar el modelo en formato SavedModel
export_path = 'linear-model/1/'

# Guarda el modelo en el directorio especificado
tf.saved_model.save(linear_model, os.path.join('./', export_path))