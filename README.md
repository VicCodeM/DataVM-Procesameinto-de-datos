# DataVM-Procesameinto de datos
# Librería de Limpieza y Procesamiento de Datos

Esta librería proporciona una serie de funciones para la limpieza, procesamiento y análisis de datos. Además, incluye métodos para interactuar con bases de datos como SQLite, MySQL, SQL Server, PostgreSQL y Oracle.

## Instalación

Asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy mysql-connector-python pyodbc psycopg2 cx_Oracle
```

## Uso

### Cargar y Guardar Datos

```python
# Cargar datos desde un archivo CSV o Excel
df = cargar_datos('ruta_archivo.csv')

# Guardar datos en un archivo CSV o Excel
guardar_datos(df, 'ruta_archivo.csv')
```

### Manejo de Valores Nulos

```python
# Verificar valores nulos
verificar_nulos(df)

# Eliminar filas con valores nulos
df = eliminar_nulos(df)

# Rellenar valores nulos
df = rellenar_nulos(df, metodo="media")
```

### Manejo de Datos Duplicados

```python
# Verificar duplicados
verificar_duplicados(df)

# Eliminar duplicados
df = eliminar_duplicados(df)
```

### Manejo de Outliers

```python
# Detectar outliers
detectar_outliers(df, 'columna', metodo="zscore")

# Eliminar outliers
df = eliminar_outliers(df, 'columna', metodo="zscore")
```

### Codificación de Variables Categóricas

```python
# Codificar variables categóricas
df = codificar_categoricas(df, ['columna_categorica'], metodo="onehot")
```

### Escalado de Datos

```python
# Escalar datos
df = escalar_datos(df, ['columna_numerica'], metodo="standard")
```

### Selección de Características

```python
# Seleccionar características
seleccionar_caracteristicas(df, 'target', k=10)
```

### Transformación de Datos

```python
# Transformar datos
df = transformar_datos(df, 'columna', metodo="log")
```

### Validación Cruzada

```python
# Validar modelo
validar_modelo(modelo, X, y, cv=5)
```

### Manejo de Datos Desbalanceados

```python
# Balancear datos
X_resampled, y_resampled = balancear_datos(X, y)
```

### Guardado de Modelos

```python
# Guardar modelo
guardar_modelo(modelo, 'modelo.pkl')

# Cargar modelo
modelo = cargar_modelo('modelo.pkl')
```

### Interacción con Bases de Datos

#### SQLite

```python
# Guardar datos en SQLite
guardar_en_sqlite('archivo.csv', 'tabla', 'basedatos.db')

# Cargar datos desde SQLite
df = cargar_desde_sqlite('basedatos.db', 'tabla')

# Insertar datos en SQLite
insertar_en_sqlite(df, 'basedatos.db', 'tabla')

# Actualizar registros en SQLite
actualizar_en_sqlite('basedatos.db', 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en SQLite
eliminar_en_sqlite('basedatos.db', 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en SQLite
resultados = ejecutar_consulta_sqlite('basedatos.db', "SELECT * FROM tabla WHERE columna = 'condicion'")
```

#### MySQL

```python
# Conectar a MySQL
conn_mysql = conectar_mysql('host', 'usuario', 'contraseña', 'basedatos')

# Cargar datos desde MySQL
df = cargar_desde_mysql(conn_mysql, 'tabla')

# Insertar datos en MySQL
insertar_en_mysql(conn_mysql, df, 'tabla')

# Actualizar registros en MySQL
actualizar_en_mysql(conn_mysql, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en MySQL
eliminar_en_mysql(conn_mysql, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en MySQL
resultados = ejecutar_consulta_mysql(conn_mysql, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión MySQL
cerrar_conexion_mysql(conn_mysql)
```

#### SQL Server

```python
# Conectar a SQL Server
conn_sql_server = conectar_sql_server('server', 'database', 'username', 'password')

# Cargar datos desde SQL Server
df = cargar_desde_sql_server(conn_sql_server, 'tabla')

# Insertar datos en SQL Server
insertar_en_sql_server(conn_sql_server, df, 'tabla')

# Actualizar registros en SQL Server
actualizar_en_sql_server(conn_sql_server, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en SQL Server
eliminar_en_sql_server(conn_sql_server, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en SQL Server
resultados = ejecutar_consulta_sql_server(conn_sql_server, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión SQL Server
cerrar_conexion_sql_server(conn_sql_server)
```

#### PostgreSQL

```python
# Conectar a PostgreSQL
conn_postgresql = conectar_postgresql('host', 'usuario', 'contraseña', 'basedatos')

# Cargar datos desde PostgreSQL
df = cargar_desde_postgresql(conn_postgresql, 'tabla')

# Insertar datos en PostgreSQL
insertar_en_postgresql(conn_postgresql, df, 'tabla')

# Actualizar registros en PostgreSQL
actualizar_en_postgresql(conn_postgresql, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en PostgreSQL
eliminar_en_postgresql(conn_postgresql, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en PostgreSQL
resultados = ejecutar_consulta_postgresql(conn_postgresql, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión PostgreSQL
cerrar_conexion_postgresql(conn_postgresql)
```

#### Oracle

```python
# Conectar a Oracle
conn_oracle = conectar_oracle('username', 'password', 'dsn')

# Cargar datos desde Oracle
df = cargar_desde_oracle(conn_oracle, 'tabla')

# Insertar datos en Oracle
insertar_en_oracle(conn_oracle, df, 'tabla')

# Actualizar registros en Oracle
actualizar_en_oracle(conn_oracle, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en Oracle
eliminar_en_oracle(conn_oracle, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en Oracle
resultados = ejecutar_consulta_oracle(conn_oracle, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión Oracle
cerrar_conexion_oracle(conn_oracle)
```

## Ejemplo Completo

```python
# Cargar datos desde un archivo CSV
df = cargar_datos('datos.csv')

# Verificar y eliminar nulos
verificar_nulos(df)
df = eliminar_nulos(df)

# Verificar y eliminar duplicados
verificar_duplicados(df)
df = eliminar_duplicados(df)

# Detectar y eliminar outliers
detectar_outliers(df, 'columna', metodo="zscore")
df = eliminar_outliers(df, 'columna', metodo="zscore")

# Codificar variables categóricas
df = codificar_categoricas(df, ['columna_categorica'], metodo="onehot")

# Escalar datos
df = escalar_datos(df, ['columna_numerica'], metodo="standard")

# Seleccionar características
seleccionar_caracteristicas(df, 'target', k=10)

# Transformar datos
df = transformar_datos(df, 'columna', metodo="log")

# Validar modelo
modelo = RandomForestClassifier()
validar_modelo(modelo, X, y, cv=5)

# Balancear datos
X_resampled, y_resampled = balancear_datos(X, y)

# Guardar modelo
guardar_modelo(modelo, 'modelo.pkl')

# Cargar modelo
modelo = cargar_modelo('modelo.pkl')

# Guardar datos en SQLite
guardar_en_sqlite('datos.csv', 'tabla', 'basedatos.db')

# Cargar datos desde SQLite
df = cargar_desde_sqlite('basedatos.db', 'tabla')

# Insertar datos en SQLite
insertar_en_sqlite(df, 'basedatos.db', 'tabla')

# Actualizar registros en SQLite
actualizar_en_sqlite('basedatos.db', 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en SQLite
eliminar_en_sqlite('basedatos.db', 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en SQLite
resultados = ejecutar_consulta_sqlite('basedatos.db', "SELECT * FROM tabla WHERE columna = 'condicion'")

# Conectar a MySQL
conn_mysql = conectar_mysql('host', 'usuario', 'contraseña', 'basedatos')

# Cargar datos desde MySQL
df = cargar_desde_mysql(conn_mysql, 'tabla')

# Insertar datos en MySQL
insertar_en_mysql(conn_mysql, df, 'tabla')

# Actualizar registros en MySQL
actualizar_en_mysql(conn_mysql, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en MySQL
eliminar_en_mysql(conn_mysql, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en MySQL
resultados = ejecutar_consulta_mysql(conn_mysql, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión MySQL
cerrar_conexion_mysql(conn_mysql)

# Conectar a SQL Server
conn_sql_server = conectar_sql_server('server', 'database', 'username', 'password')

# Cargar datos desde SQL Server
df = cargar_desde_sql_server(conn_sql_server, 'tabla')

# Insertar datos en SQL Server
insertar_en_sql_server(conn_sql_server, df, 'tabla')

# Actualizar registros en SQL Server
actualizar_en_sql_server(conn_sql_server, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en SQL Server
eliminar_en_sql_server(conn_sql_server, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en SQL Server
resultados = ejecutar_consulta_sql_server(conn_sql_server, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión SQL Server
cerrar_conexion_sql_server(conn_sql_server)

# Conectar a PostgreSQL
conn_postgresql = conectar_postgresql('host', 'usuario', 'contraseña', 'basedatos')

# Cargar datos desde PostgreSQL
df = cargar_desde_postgresql(conn_postgresql, 'tabla')

# Insertar datos en PostgreSQL
insertar_en_postgresql(conn_postgresql, df, 'tabla')

# Actualizar registros en PostgreSQL
actualizar_en_postgresql(conn_postgresql, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en PostgreSQL
eliminar_en_postgresql(conn_postgresql, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en PostgreSQL
resultados = ejecutar_consulta_postgresql(conn_postgresql, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión PostgreSQL
cerrar_conexion_postgresql(conn_postgresql)

# Conectar a Oracle
conn_oracle = conectar_oracle('username', 'password', 'dsn')

# Cargar datos desde Oracle
df = cargar_desde_oracle(conn_oracle, 'tabla')

# Insertar datos en Oracle
insertar_en_oracle(conn_oracle, df, 'tabla')

# Actualizar registros en Oracle
actualizar_en_oracle(conn_oracle, 'tabla', "columna = 'nuevo_valor'", "columna = 'condicion'")

# Eliminar registros en Oracle
eliminar_en_oracle(conn_oracle, 'tabla', "columna = 'valor_a_eliminar'")

# Ejecutar consultas SQL personalizadas en Oracle
resultados = ejecutar_consulta_oracle(conn_oracle, "SELECT * FROM tabla WHERE columna = 'condicion'")

# Cerrar conexión Oracle
cerrar_conexion_oracle(conn_oracle)
```


## Uso

### Importación de Funciones y Clases

Puedes importar funciones y clases desde el paquete `data_processing` de la siguiente manera:

```python
from data_processing import (
    cargar_datos,
    eliminar_nulos,
    construir_modelo,
    compilar_modelo,
    entrenar_modelo,
    evaluar_modelo
)
```

### Ejemplo Completo

```python
from data_processing import (
    cargar_datos,
    eliminar_nulos,
    construir_modelo,
    compilar_modelo,
    entrenar_modelo,
    evaluar_modelo,
    preprocesar_datos
)



```bash
pip install -r requirements.txt
```

### Uso

#### Preprocesamiento de Datos

```python
from data_processing import neural_networks

# Preprocesar datos para redes neuronales
X_train, X_test, y_train, y_test = neural_networks.preprocesar_datos(X, y)
```

#### Preprocesamiento de Datos para NLP

```python
# Preprocesar datos para NLP
X_train, X_test, y_train, y_test, word_index = neural_networks.preprocesar_nlp(texts, labels)
```

#### Preprocesamiento de Datos para GPT-2 y GPT-3

```python
# Preprocesar datos para GPT-2 y GPT-3
inputs = neural_networks.preprocesar_gpt(texts)
```

#### Construir y Compilar el Modelo

```python
# Construir el modelo
model = neural_networks.construir_modelo(input_dim=X_train.shape[1], output_dim=len(y.unique()))

# Compilar el modelo
model = neural_networks.compilar_modelo(model)
```

#### Entrenar el Modelo

```python
# Callbacks para el entrenamiento
callbacks_list = neural_networks.callbacks()

# Entrenar el modelo
history = neural_networks.entrenar_modelo(model, X_train, y_train, callbacks=callbacks_list)
```

#### Visualizar Métricas de Entrenamiento

```python
# Visualizar métricas de entrenamiento
neural_networks.visualizar_metricas(history)
```

#### Evaluar el Modelo

```python
# Evaluar el modelo
loss, accuracy = neural_networks.evaluar_modelo(model, X_test, y_test)
```

#### Guardar y Cargar el Modelo

```python
# Guardar el modelo
neural_networks.guardar_modelo(model, 'modelo.h5')

# Cargar el modelo
model = neural_networks.cargar_modelo('modelo.h5')
```

#### Predecir con el Modelo

```python
# Predecir con el modelo
predictions = neural_networks.predecir(model, X_test)
```

### Ejemplo Completo

```python
from data_processing import data_processing, neural_networks

# Cargar y preprocesar datos
df = data_processing.cargar_datos('datos.csv')
df = data_processing.eliminar_nulos(df)
df = data_processing.eliminar_duplicados(df)

# Preprocesar datos para redes neuronales
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = neural_networks.preprocesar_datos(X, y)

# Construir y compilar el modelo
model = neural_networks.construir_modelo(input_dim=X_train.shape[1], output_dim=len(y.unique()))
model = neural_networks.compilar_modelo(model)

# Callbacks para el entrenamiento
callbacks_list = neural_networks.callbacks()

# Entrenar el modelo
history = neural_networks.entrenar_modelo(model, X_train, y_train, callbacks=callbacks_list)

# Visualizar métricas de entrenamiento
neural_networks.visualizar_metricas(history)

# Evaluar el modelo
loss, accuracy = neural_networks.evaluar_modelo(model, X_test, y_test)

# Guardar el modelo
neural_networks.guardar_modelo(model, 'modelo.h5')

# Cargar el modelo
model = neural_networks.cargar_modelo('modelo.h5')

# Predecir con el modelo
predictions = neural_networks.predecir(model, X_test)
```

### Ejemplo de Uso para NLP

```python
from data_processing import data_processing, neural_networks

# Cargar y preprocesar datos
texts = ["ejemplo de texto 1", "ejemplo de texto 2", "ejemplo de texto 3"]
labels = [0, 1, 0]

# Preprocesar datos para NLP
X_train, X_test, y_train, y_test, word_index = neural_networks.preprocesar_nlp(texts, labels)

# Construir y compilar el modelo
model = neural_networks.construir_modelo(input_dim=(X_train.shape[1],), output_dim=len(np.unique(labels)), model_type='nlp', nlp_vocab_size=len(word_index) + 1, nlp_max_length=X_train.shape[1])
model = neural_networks.compilar_modelo(model)

# Callbacks para el entrenamiento
callbacks_list = neural_networks.callbacks()

# Entrenar el modelo
history = neural_networks.entrenar_modelo(model, X_train, y_train, callbacks=callbacks_list)

# Visualizar métricas de entrenamiento
neural_networks.visualizar_metricas(history)

# Evaluar el modelo
loss, accuracy = neural_networks.evaluar_modelo(model, X_test, y_test)

# Guardar el modelo
neural_networks.guardar_modelo(model, 'modelo_nlp.h5')

# Cargar el modelo
model = neural_networks.cargar_modelo('modelo_nlp.h5')

# Predecir con el modelo
predictions = neural_networks.predecir(model, X_test)
```

### Ejemplo de Uso para GPT-2

```python
from data_processing import data_processing, neural_networks

# Cargar y preprocesar datos
texts = ["ejemplo de texto 1", "ejemplo de texto 2", "ejemplo de texto 3"]
labels = [0, 1, 0]

# Preprocesar datos para GPT-2
inputs = neural_networks.preprocesar_gpt(texts)

# Construir el modelo GPT-2
model = neural_networks.construir_gpt2(output_dim=len(np.unique(labels)))

# Compilar el modelo
model = neural_networks.compilar_modelo(model)

# Callbacks para el entrenamiento
callbacks_list = neural_networks.callbacks()

# Entrenar el modelo
history = neural_networks.entrenar_modelo(model, inputs['input_ids'], labels, callbacks=callbacks_list)

# Visualizar métricas de entrenamiento
neural_networks.visualizar_metricas(history)

# Evaluar el modelo
loss, accuracy = neural_networks.evaluar_modelo(model, inputs['input_ids'], labels)

# Guardar el modelo
neural_networks.guardar_modelo(model, 'modelo_gpt2.h5')

# Cargar el modelo
model = neural_networks.cargar_modelo('modelo_gpt2.h5')

# Predecir con el modelo
predictions = neural_networks.predecir(model, inputs['input_ids'])
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para cualquier mejora o corrección.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
```
