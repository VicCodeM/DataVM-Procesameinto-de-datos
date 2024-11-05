import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Embedding, GRU, Bidirectional, SimpleRNN
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TFAutoModelForSequenceClassification

# Construir un modelo de red neuronal
def construir_modelo(input_dim, output_dim, hidden_layers=[64, 32], dropout_rate=0.2, model_type='dense', nlp_vocab_size=None, nlp_max_length=None):
    model = Sequential()
    
    if model_type == 'dense':
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
    elif model_type == 'cnn':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(hidden_layers[0], activation='relu'))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
    elif model_type == 'lstm':
        model.add(LSTM(hidden_layers[0], input_shape=input_dim))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
    elif model_type == 'gru':
        model.add(GRU(hidden_layers[0], input_shape=input_dim))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
    elif model_type == 'rnn':
        model.add(SimpleRNN(hidden_layers[0], input_shape=input_dim))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
    elif model_type == 'nlp':
        model.add(Embedding(nlp_vocab_size, 128, input_length=nlp_max_length))
        model.add(LSTM(hidden_layers[0]))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_dim, activation='softmax'))
    return model

# Compilar el modelo
def compilar_modelo(model, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', metrics=['accuracy']):
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer no soportado")
    
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model

# Entrenar el modelo
def entrenar_modelo(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=None):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks)
    return history

# Evaluar el modelo
def evaluar_modelo(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Pérdida: {loss}, Precisión: {accuracy}")
    return loss, accuracy

# Preprocesamiento de datos para redes neuronales
def preprocesar_datos(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    
    return X_train, X_test, y_train, y_test

# Preprocesamiento de datos para NLP
def preprocesar_nlp(texts, labels, max_words=10000, max_length=100, test_size=0.2, random_state=42):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_length)
    
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    
    return X_train, X_test, y_train, y_test, tokenizer.word_index

# Preprocesamiento de datos para GPT-2 y GPT-3
def preprocesar_gpt(texts, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer(texts, return_tensors='tf', max_length=max_length, truncation=True, padding='max_length')
    return inputs

# Construir un modelo GPT-2
def construir_gpt2(output_dim):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(GPT2Tokenizer.from_pretrained('gpt2')))
    model.lm_head = Dense(output_dim, activation='softmax')(model.lm_head)
    return model

# Construir un modelo GPT-3 (usando un modelo preentrenado de Hugging Face)
def construir_gpt3(output_dim):
    model = TFAutoModelForSequenceClassification.from_pretrained('gpt-3', num_labels=output_dim)
    return model

# Visualizar métricas de entrenamiento
def visualizar_metricas(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.show()

# Guardar el modelo
def guardar_modelo(model, ruta_archivo):
    model.save(ruta_archivo)
    print(f"Modelo guardado en {ruta_archivo}")

# Cargar el modelo
def cargar_modelo(ruta_archivo):
    model = load_model(ruta_archivo)
    print(f"Modelo cargado desde {ruta_archivo}")
    return model

# Predecir con el modelo
def predecir(model, X):
    return model.predict(X)

# Callbacks para el entrenamiento
def callbacks(monitor='val_loss', patience=10, save_best_only=True, save_path='best_model.h5'):
    early_stopping = EarlyStopping(monitor=monitor, patience=patience)
    model_checkpoint = ModelCheckpoint(save_path, monitor=monitor, save_best_only=save_best_only)
    return [early_stopping, model_checkpoint]