# Face_AI

Proyecto de reconocimiento facial para la clasificación de emociones y personas usando redes neuronales convolucionales (CNN) en Keras/TensorFlow.

## Descripción

Este proyecto permite entrenar un modelo capaz de identificar la emoción y la persona en imágenes faciales. El modelo es multi-salida: predice tanto la emoción (7 clases) como la persona (2 clases). Incluye scripts para entrenamiento, pruebas individuales (GUI) y pruebas por lotes.

## Estructura del Proyecto

- `train_model.py`: Entrena el modelo CNN con imágenes organizadas por carpetas de emociones.
- `test.py`: Interfaz gráfica (GUI) para probar el modelo con imágenes individuales.
- `test_colab.py`: Script para probar el modelo con todas las imágenes de una carpeta.
- `images/`: Carpeta con subcarpetas por emoción, cada una con imágenes nombradas por persona.
- `graphics/`: Carpeta donde se guardan las gráficas y resultados del entrenamiento.
- `model_emotions.keras`: Archivo del modelo entrenado (se genera tras el entrenamiento).

## Requisitos

- Python 3.8+
- TensorFlow y Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Pillow
- scikit-learn
- customtkinter (solo para la GUI)

Instala los requisitos con:

```bash
pip install tensorflow keras opencv-python numpy matplotlib seaborn pillow scikit-learn customtkinter
```

## Entrenamiento

1. Organiza tus imágenes en la carpeta `images/`, con subcarpetas para cada emoción (`alegre`, `cansado`, etc.).
2. Nombra los archivos de imagen como `Magleo_*.jpg` o `Hector_*.jpg` según la persona.
3. Ejecuta el script de entrenamiento:

```bash
python train_model.py
```

Se generarán el modelo entrenado y gráficas en la carpeta `graphics/`.

## Prueba con Interfaz Gráfica

Ejecuta:

```bash
python test.py
```

Carga una imagen y el sistema detectará la emoción y la persona.

## Prueba por Lote (google colab)

Coloca imágenes en la carpeta `testing/` y ejecuta:

```bash
python test_colab.py
```

El script mostrará las predicciones para cada imagen.

## Notas

- El modelo requiere que las imágenes tengan un rostro visible y estén correctamente nombradas.
- El umbral de confianza para la identificación de persona puede ajustarse en `test.py`.


