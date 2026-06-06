##  Instrucciones de Ejecución

### Requisitos previos

- Cuenta en **Google Colab**: https://colab.research.google.com
- Cuenta en **Kaggle**: https://www.kaggle.com
- Token de API de Kaggle

---

### Paso 1 — Abrir el notebook en Google Colab

1. Ve a https://colab.research.google.com
2. Selecciona `File → Open notebook → GitHub`
3. Pega la URL del repositorio:
   ```
   https://github.com/DavidMoraV/Tarea_2_computacion_paralela
   ```
4. Selecciona el archivo `Tarea2.ipynb`

---

### Paso 2 — Activar la GPU

**Obligatorio antes de ejecutar cualquier celda:**

```
Runtime → Change Runtime Type → Hardware Accelerator → GPU (T4)
```

Verificar que la GPU esté activa con la celda 1.1 (`nvidia-smi`).

---

### Paso 3 — Configurar credenciales de Kaggle

En la celda 1.4 del notebook, completar con tus datos:

```python
KAGGLE_USERNAME = "tu_usuario_kaggle"   # tu usuario en kaggle.com
KAGGLE_TOKEN    = "KGAT_..."            # token desde kaggle.com/settings → Tokens de API
```

**Para obtener el token:**
1. Ve a https://www.kaggle.com/settings/tokens
2. Clic en **"Generar nuevo token"**
3. Copia el token generado (`KGAT_...`)

---

### Paso 4 — Ejecutar los pasos en orden

El notebook está organizado en 7 pasos que deben ejecutarse secuencialmente en la **misma sesión de Colab**:

| Paso | Archivo / Sección | Descripción | Tiempo aprox. |
|------|-------------------|-------------|---------------|
| 1 | Celdas 1–15 | Configuración del entorno + descarga del dataset | 3–5 min |
| 2 | Celdas 17–33 | Preprocesamiento + pipelines `tf.data` | 2–3 min |
| 3 | Celdas 35–49 | Modelo CNN Flax NNX + entrenamiento base (20 épocas) | 5–10 min |
| 4 | Celdas 51–65 | Experimentos batch / lr / tamaño de red | 35–45 min |
| 5 | Celdas 67–72 | Grid Search de hiperparámetros (5 configs × 15–30 épocas) | 30–40 min |
| 6 | Celdas 74–81 | Análisis de precisión numérica float32 / float16 / bfloat16 | 15–20 min |
| 7 | Celdas 83–84 | Análisis y discusión — 9 preguntas | < 1 min |

> ⚠️ **No reiniciar la sesión** entre pasos — las variables del Paso 1 y 2 son necesarias para todos los pasos siguientes.

---

### Paso 5 — Reproducir experimentos individuales

Para reproducir un experimento específico sin correr todo el notebook, ejecutar primero las celdas de configuración obligatorias:

```
Celdas 1–15  →  Entorno, dataset y constantes globales
Celdas 17–33 →  Preprocesamiento y pipelines
Celdas 35–41 →  Clase CNN, función de pérdida y optimizador
Celdas 51    →  Función ejecutar_experimento()
```

Luego ejecutar solo el experimento deseado.

---

### Paso 6 — Regenerar las gráficas

Todas las gráficas se generan automáticamente al ejecutar las celdas correspondientes. Para regenerarlas sin reentrenar el modelo, los historiales de entrenamiento quedan guardados en las variables:

```python
historial_base       # Paso 3 — entrenamiento base
resultados_batch     # Experimento 6.1 — tamaño de lote
resultados_lr        # Experimento 6.2 — learning rate
resultados_red       # Experimento 6.3 — tamaño de red
resultados_grid      # Paso 5 — Grid Search
resultados_dtype     # Paso 6 — precisión numérica
```

---

### Tiempos totales estimados

| Acelerador | Tiempo total completo |
|------------|----------------------|
| Tesla T4 (Colab gratuito) | ~90–120 min |
| A100 (Colab Pro) | ~25–35 min |

---

### Notas importantes

- **Sesión de Colab:** la sesión gratuita se desconecta tras ~90 min de inactividad. Mantener la pestaña activa o usar Colab Pro para sesiones más largas.
- **Dataset:** se descarga automáticamente desde Kaggle (~150 MB). Si la descarga falla, verificar que el token de Kaggle tenga permisos de descarga del dataset.
- **Reproducibilidad:** todos los experimentos usan `SEMILLA = 42` para garantizar resultados reproducibles entre ejecuciones.
- **Memoria GPU:** el pipeline usa `tf.data` con `prefetch AUTOTUNE` — si aparece un error de OOM (Out of Memory), reducir `BATCH_SIZE` a 16.




##  Análisis y Discusión

### 1. ¿Qué ventajas ofrece Flax NNX respecto a implementar modelos directamente en JAX?

Flax NNX proporciona una capa de abstracción orientada a objetos sobre JAX puro que simplifica la construcción de modelos. Las ventajas observadas durante la implementación:

- **Gestión automática del estado:** `nnx.Module` administra internamente parámetros, estadísticas de `BatchNorm` y estados de `Dropout`, eliminando la necesidad de manejar manualmente pytrees de parámetros.
- **Componentes listos para usar:** `nnx.Conv`, `nnx.Linear`, `nnx.BatchNorm` y `nnx.Dropout` encapsulan inicialización, forward pass y actualización de estado en una sola clase.
- **Integración natural con `@nnx.jit`:** el decorador maneja automáticamente la separación entre estado mutable y funciones puras requerida por JAX.
- En JAX puro habría sido necesario gestionar manualmente los diccionarios de parámetros, aplicar `jax.grad` sobre funciones que los reciben como argumento, y mantener el estado de `BatchNorm` por separado. Flax NNX abstrae toda esa complejidad.

---

### 2. ¿Qué beneficios aporta la compilación mediante `jax.jit`?

Los beneficios observados directamente durante el entrenamiento:

- **Reducción de tiempo tras la primera época:** la época 1 tomó ~48 s (compilación XLA), mientras las épocas siguientes tomaron ~9-12 s — una mejora de **4-5×**.
- **Fusión de operaciones:** XLA fusiona operaciones consecutivas (`Conv → BatchNorm → ReLU`) en un único kernel de GPU, reduciendo transferencias de memoria intermedias.
- **Throughput sostenido:** entre 225 y 294 img/s en Tesla T4 durante todo el entrenamiento.
- **Eliminación del overhead de Python:** una vez compilado, el paso de entrenamiento se ejecuta completamente en GPU sin intervención del intérprete.

---

### 3. ¿Qué hiperparámetro tuvo mayor impacto sobre la exactitud final?

El **número de épocas** tuvo el mayor impacto positivo:

| Configuración | Épocas | Exactitud prueba |
|--------------|--------|-----------------|
| Modelo base  | 20     | 70.23%          |
| Config E     | 30     | **75.16%**      |

El **learning rate** tuvo el mayor impacto negativo cuando era inadecuado:

| Learning Rate | Exactitud prueba |
|--------------|-----------------|
| `lr = 1e-2`  | 18.42% — no convergió |
| `lr = 1e-3`  | 68.91% — convergencia normal |
| `lr = 1e-4`  | 68.59% — convergencia lenta |

---

### 4. ¿Qué hiperparámetro tuvo mayor impacto sobre el tiempo de entrenamiento?

El **batch size** fue el más influyente:

| Batch Size | Tiempo/época |
|------------|-------------|
| 16         | 10.8 s      |
| 32         | 9.4 s       |
| 128        | 10.9 s      |

El tamaño de la red tuvo impacto moderado: pasar de 1M a 16M de parámetros solo añadió +0.7 s/época, gracias al paralelismo masivo de la GPU.

---

### 5. ¿Cómo afectó el tamaño de lote al rendimiento observado?

| Batch Size | Acc. Prueba | Throughput  |
|------------|-------------|-------------|
| 32         | **70.72%**  | 286 img/s   |
| 64         | 60.00%      | 282 img/s   |
| 128        | 48.76%      | 247 img/s   |

Los lotes pequeños producen gradientes con mayor varianza, lo que actúa como **regularizador implícito** y ayuda a escapar mínimos locales. Con `batch=128` el modelo converge a soluciones menos generalizables dado el tamaño reducido del dataset (3,846 imágenes).

---

### 6. ¿Qué diferencias encontró entre `float32`, `float16` y `bfloat16`?

| dtype      | Acc. Prueba | Estabilidad  | Tiempo/época | Memoria |
|------------|-------------|--------------|-------------|---------|
| `float32`  | **75.16%**  |  Estable     | 9.7 s       | 8.4 MB  |
| `float16`  | 73.85%      |  Estable     | 9.9 s       | 4.2 MB  |
| `bfloat16` | 66.94%      |  Estable     | 10.1 s      | 4.2 MB  |

- Los tres formatos fueron **numéricamente estables** en la Tesla T4.
- `float16` reduce la memoria a la mitad con solo −1.31 pp de exactitud respecto a `float32`.
- `bfloat16` fue el menos preciso pese a tener mayor rango dinámico — su menor precisión de mantisa (7 bits vs 23 en `float32`) afectó la calidad de los gradientes.

---

### 7. ¿Cuál configuración produjo el mejor balance entre rendimiento y exactitud?

**Configuración E — `batch=32 · lr=1e-3 · filtros=32 · neuronas=256 · 30 épocas · float32`**

| Métrica           | Valor      |
|-------------------|------------|
| Exactitud prueba  | **75.16%** |
| Throughput        | 294 img/s  |
| Tiempo/época      | 9.1 s      |
| Memoria           | 8.4 MB     |

Si se prioriza eficiencia de memoria: `float16` con la misma configuración logra ~73.85% con solo 4.2 MB — **ahorra 50% de memoria con −1.31 pp de exactitud**.

---

### 8. ¿Qué limitaciones encontró al utilizar Google Colab?

- **Tiempo de sesión:** Colab desconecta la sesión tras ~90 min de inactividad o ~12 hrs de uso continuo, lo que obligó a re-ejecutar el pipeline completo en ocasiones.
- **RAM limitada:** ~12 GB de RAM CPU. Pipelines con `batch_size=16` y `prefetch AUTOTUNE` saturaron la memoria en algunos experimentos.
- **GPU no garantizada:** en horas pico no siempre se asignó Tesla T4, variando los tiempos entre ejecuciones.
- **Sin persistencia de variables:** al reiniciar la sesión se pierden todos los modelos entrenados y resultados intermedios — sin checkpoints propios, hay que reentrenar desde cero.
- **Disco efímero:** los archivos en `/content` desaparecen al reiniciar. Se requiere Google Drive o re-descarga del dataset en cada sesión.

---

### 9. ¿Qué mejoras futuras propondría para aumentar el desempeño del modelo?

1. **Transfer learning:** usar una red preentrenada (ResNet50, EfficientNet-B0, ViT) como feature extractor y entrenar solo las capas finales — se estima >90% de exactitud con el mismo dataset.
2. **Aumento de datos más agresivo:** rotaciones ±30°, zoom aleatorio, recortes y técnicas como MixUp o CutMix mejorarían la generalización dado el tamaño reducido del dataset.
3. **Learning rate scheduling avanzado:** warm-up + cosine annealing con reinicios (SGDR) para escapar mejor los mínimos locales en las épocas finales.
4. **Regularización adicional:** L2 weight decay en capas densas y label smoothing en la función de pérdida para reducir sobreconfianza.
5. **Arquitectura residual (ResNet-style):** conexiones skip permitirían entrenar redes más profundas sin degradación del gradiente.
6. **GPU dedicada (A100/V100):** el throughput escalaría de ~294 img/s a >2,000 img/s usando `bfloat16` y `batch_size=256`.
