import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Cargamos y procesamos las imágenes
# Ruta a las imágenes y al archivo CSV
image_dir = 'img/Training_Set/Training_Set/Training'  # Ajuste en la ruta de Windows a Unix-like
csv_file = 'img/Training_Set/Training_Set/RFMiD_Training_Labels.csv'

image_names = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Cargar el archivo CSV
df = pd.read_csv(csv_file)
label_dict = dict(zip(df['ID'].astype(str) + '.png', df['Disease_Risk']))

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Redimensiono a 224x224 píxeles
    image = image / 255.0  # Normalizo
    return image

images = []
labels = []
for image_name in os.listdir(image_dir):
    if image_name.endswith('.png'):
        image_path = os.path.join(image_dir, image_name)
        image = load_and_preprocess_image(image_path)
        images.append(image)
        labels.append(label_dict[image_name])

# Convertir a arrays de NumPy
images = np.array(images)
labels = np.array(labels)

# Aumentamos datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(images)

# Escalar los datos
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images.reshape(len(images), -1))

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=100)  # Incrementar el número de componentes
images_pca = pca.fit_transform(images_scaled)

# Dividir el dataset en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Modelo preentrenado para extracción de características
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-4]:  # Descongelar las últimas 4 capas
    layer.trainable = False

# Extraer características con el modelo preentrenado
features = base_model.predict(images)

# Aplanar las características extraídas y concatenar con características PCA
features_flattened = features.reshape((features.shape[0], -1))
combined_features = np.concatenate((features_flattened, images_pca), axis=1)

# Dividir el dataset con las nuevas características
X_train_combined, X_temp_combined, y_train_combined, y_temp_combined = train_test_split(combined_features, labels, test_size=0.3, random_state=42)
X_val_combined, X_test_combined, y_val_combined, y_test_combined = train_test_split(X_temp_combined, y_temp_combined, test_size=0.5, random_state=42)

# Modelo preentrenado para extracción de características
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-4]:  # Descongelar las últimas 4 capas
    layer.trainable = False

# Extraer características con el modelo preentrenado
features = base_model.predict(images)

# Aplanar las características extraídas y concatenar con características PCA
features_flattened = features.reshape((features.shape[0], -1))
combined_features = np.concatenate((features_flattened, images_pca), axis=1)

# Dividir el dataset con las nuevas características
X_train_combined, X_temp_combined, y_train_combined, y_temp_combined = train_test_split(combined_features, labels, test_size=0.3, random_state=42)
X_val_combined, X_test_combined, y_val_combined, y_test_combined = train_test_split(X_temp_combined, y_temp_combined, test_size=0.5, random_state=42)


input_shape = combined_features.shape[1]
model = Sequential([
    Input(shape=(input_shape,)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # Regularización L2
    Dropout(0.5),
    Dense(46, activation='softmax')  # 46 clases para las diferentes enfermedades
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Entrenar el modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train_combined, y_train_combined, epochs=20, batch_size=32, validation_data=(X_val_combined, y_val_combined), callbacks=[early_stopping])

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test_combined, y_test_combined)
print(f'Test accuracy: {test_acc}')

# Realizar predicciones en el conjunto de test
predictions = model.predict(X_test_combined)
predicted_classes = np.argmax(predictions, axis=1)

# Evaluar las predicciones
cm = confusion_matrix(y_test_combined, predicted_classes)
print("Matriz de Confusión:")
print(cm)

print("Reporte de Clasificación:")
print(classification_report(y_test_combined, predicted_classes))

# Guardar el modelo
model.save('modeloretinaentrenado.keras')