#  Clasificación de Patologías Médicas (NHANES) mediante Deep Learning
INTEGRANTES:
# INTEGRANTES: ABALOS PEREZ JUAN JOSE.
# MORA BARRIONUEVO NELVA ADALIT

Este repositorio contiene la implementación de una Red Neuronal Profunda (MLP) desarrollada en PyTorch para la clasificación multiclase de condiciones de salud crónicas utilizando el dataset oficial NHANES (National Health and Nutrition Examination Survey).

El objetivo principal de este proyecto es predecir si un paciente padece **Obesidad, Hipertensión o Colesterol Alto** basándose en 34 variables clínicas y sociodemográficas, aplicando rigurosas buenas prácticas de Machine Learning para evitar sesgos y sobreajustes (Overfitting).

---

##  Arquitectura y Tecnologías
- **Framework Principal:** PyTorch (con soporte para aceleración por hardware CUDA).
- **Preprocesamiento:** Scikit-Learn (`StandardScaler`, `LabelEncoder`), Pandas, Numpy.
- **Visualización:** Matplotlib, Seaborn.
- **Arquitectura de la Red:** Perceptrón Multicapa (MLP) de 3 capas densas (Input: 34 -> Oculta: 256 -> Oculta: 128 -> Output: 3).

---

##  Buenas Prácticas Implementadas

Este proyecto fue desarrollado bajo estrictas guías de rigor académico y matemático para garantizar que la red neuronal sea robusta y aplicable a un entorno de diagnóstico real.

### 1. Tratamiento Ético del Desbalance de Datos (Class Weights)
El dataset original presentaba un claro desbalance poblacional (ej. mucha prevalencia de Obesidad frente a Hipertensión). 
En un contexto de salud clínico, **se descartó el uso de técnicas generadoras de datos sintéticos (como SMOTE)**, ya que inventar historiales de pacientes puede introducir perfiles biológicamente irreales. 
En su lugar, se optó por una solución matemática: la implementación de **Pesos de Clase (Class Weights)** directamente en la función de pérdida (`CrossEntropyLoss`). Esto fuerza a la red neuronal a penalizar con mayor severidad los errores cometidos al diagnosticar las enfermedades minoritarias, obligando al modelo a prestar la misma atención a todas las patologías.

### 2. Regularización y Prevención de Overfitting
Para evitar que el modelo simplemente "memorice" la muestra de entrenamiento, se combinaron las siguientes estrategias:
* **Weight Decay (Regularización L2):** Aplicada directamente en el optimizador Adam (`weight_decay=1e-4`) para penalizar pesos (weights) desproporcionadamente grandes.
* **Dropout (0.4):** Apagado aleatorio del 40% de las neuronas durante el entrenamiento, obligando a la red a no depender de un solo biomarcador aislado.
* **Early Stopping:** Se monitoreó el *Accuracy* del conjunto de validación, deteniendo el entrenamiento y restaurando los pesos óptimos cuando el modelo comenzaba a sobreajustarse.

### 3. Optimización Dinámica
Se integró el programador **`ReduceLROnPlateau`**. Si el rendimiento del modelo se estanca durante 5 épocas, la Tasa de Aprendizaje (Learning Rate) se reduce a la mitad automáticamente. Esto permite que el algoritmo de descenso de gradiente realice ajustes mucho más finos al acercarse al mínimo global.

---

##  Evaluación y Métricas (El Enfoque Macro)
El uso del clásico *Accuracy* general puede ser muy engañoso en datasets desbalanceados (podría reportar un 90% de éxito solo por adivinar siempre la clase mayoritaria).

Para mitigar esta "trampa de la exactitud", se evalúa el modelo utilizando el **Promedio Macro (`average='macro'`)** en métricas clave:
* **Precision (Macro):** Mide la exactitud de las predicciones positivas para evitar *falsos positivos* (diagnosticar erróneamente a alguien sano).
* **Recall (Macro):** Mide la sensibilidad del modelo para evitar *falsos negativos* (enviar a casa a alguien que sí está enfermo).
* **F1-Score (Macro):** Media armónica balanceada entre Precision y Recall.

El uso del enfoque *Macro* asegura que el modelo sea evaluado con la misma severidad y transparencia para las tres enfermedades de manera equitativa.

---

## 🚀 Cómo ejecutar este proyecto

1. Clonar este repositorio.
2. Asegurarse de tener los archivos de datos (`X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`) en el mismo entorno o actualizar las rutas base en el cuaderno.
3. Instalar las dependencias necesarias:
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib seaborn
