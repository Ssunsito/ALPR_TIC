# Marco Teórico: Temas a Cubrir - Detección y OCR de Placas Vehiculares con Aprendizaje Federado

## 1. INTRODUCCIÓN Y CONTEXTO GENERAL

### 1.1 Visión por Computadora
- Definición y aplicaciones principales
- Procesamiento digital de imágenes
- Tareas clave: clasificación, detección, segmentación, OCR
- Relevancia en sistemas de transporte e inteligencia de tráfico

### 1.2 Lectura Automática de Placas (ALPR/ANPR)
- Definición y casos de uso (peajes, estacionamientos, seguridad, control de tráfico)
- Desafíos principales: variabilidad de luz, ángulos, distancia, deterioro
- Arquitectura típica: detección → localización → segmentación → OCR

### 1.3 Motivación del Proyecto
- Necesidad de sistemas descentralizados (privacidad, reducción de latencia)
- Aprendizaje federado como solución alternativa a centralización de datos
- Adaptación a hardware edge (Jetson Nano)

---

## 2. DETECCIÓN DE OBJETOS CON REDES NEURONALES PROFUNDAS

### 2.1 Redes Neuronales Convolucionales (CNN)
- Fundamentos: convolución, pooling, activaciones
- Capas convolucionales: extracción de características jerárquicas
- Batch Normalization: estabilización del entrenamiento
- Arquitecturas base: ResNet, VGG, MobileNet

### 2.2 Arquitecturas Modernas de Detección
- **Conceptos generales:** región de interés (RoI), anclajes (anchors), NMS
- **Modelos de una etapa (one-stage):**
  - YOLO (You Only Look Once): regresión directa en grid
  - SSD (Single Shot Detector)
  - Ventajas: velocidad en tiempo real
- **Modelos de dos etapas (two-stage):**
  - R-CNN y variantes (Fast R-CNN, Faster R-CNN)
  - Ventajas: precisión superior
  
### 2.3 YOLO - Detalles de Implementación
- Versiones: YOLOv3, YOLOv5, YOLOv8
- Configuración de entrada/salida
- Parámetros de umbral: confianza, IoU (Intersection over Union)
- Aplicación específica: detección de placas en video en tiempo real
- NMS (Non-Maximum Suppression): eliminación de detecciones duplicadas

### 2.4 Métricas de Evaluación
- Precisión (Precision), Recall, F1-score
- mAP (mean Average Precision) por IoU threshold
- Curvas Precisión-Recall
- Análisis de desempeño en dataset desbalanceado

---

## 3. RECONOCIMIENTO ÓPTICO DE CARACTERES (OCR)

### 3.1 OCR Tradicional vs. Deep Learning
- Métodos clásicos: template matching, segmentación manual
- Limitaciones de enfoques tradicionales
- Revolución con redes neuronales profundas
- Problemas de OCR: variabilidad de fuentes, ruido, deformación

### 3.2 Arquitectura CRNN (Convolutional Recurrent Neural Network)
- **Componente CNN:**
  - Extracción de características de imagen
  - Capas convolucionales para mapas de características
  - Reducción de dimensión espacial (pooling)
  
- **Componente RNN (LSTM/GRU):**
  - Modelado de dependencias temporales entre caracteres
  - LSTM (Long Short-Term Memory): arquitectura y ecuaciones
  - Bidireccionalidad: contexto pasado y futuro
  - Relevancia: captura orden y relaciones de caracteres

- **Arquitectura completa:**
  ```
  Imagen → CNN (features) → Reshape → BiLSTM → BiLSTM → Dense(num_chars)
  ```

### 3.3 Pérdida CTC (Connectionist Temporal Classification)
- Problema: alineación input-output desconocida
- Solución CTC: suma sobre todos los alineamientos posibles
- Ventajas: entrenamiento sin etiquetas de caracteres individuales
- Decodificación:
  - Greedy decoding (argmax simple)
  - Beam search (búsqueda óptima aproximada)
- Manejo de caracteres en blanco (blank token)

### 3.4 Preprocesamiento y Augmentación
- Normalización de imagen: escala de grises, redimensión
- Data augmentation: rotación, distorsión, ruido, cambios de iluminación
- Relevancia: robustez ante variaciones naturales

### 3.5 Entrenamiento del Modelo OCR
- Función de pérdida CTC
- Optimizadores: Adam, SGD con momentum
- Learning rate scheduling: ReduceLROnPlateau, warmup
- Early stopping y validación

---

## 4. APRENDIZAJE FEDERADO

### 4.1 Motivación y Concepto Fundamental
- Desafíos del ML centralizado:
  - Privacidad: datos no salen del dispositivo
  - Comunicación: reduce transferencia de datos
  - Latencia: reduce dependencia de servidor central
  - Regulación: cumplimiento de GDPR, normativas locales

- Aprendizaje federado: entrenar modelos sin centralizar datos

### 4.2 Algoritmo FedAvg (Federated Averaging)
- **Paso 1:** Servidor envía modelo global a clientes
- **Paso 2:** Cada cliente entrena localmente N épocas con sus datos
- **Paso 3:** Clientes envían pesos actualizados al servidor
- **Paso 4:** Servidor agrega pesos (promedio simple)
- **Paso 5:** Modelo global actualizado, regresa a Paso 1
- Ecuaciones matemáticas de agregación

### 4.3 Variantes de Agregación
- **Promedio simple (FedAvg):** todos los clientes pesan igual
- **Agregación ponderada:** según cantidad de datos locales
- **Agregación robusta:** resistencia a valores atípicos
- **Agregación comprimida:** reducir transferencia de datos

### 4.4 Desafíos en Federated Learning
- **No-IID (datos no distribuidos idénticamente):**
  - Datos heterogéneos entre clientes
  - Impacto en convergencia
  - Estrategias de mitigación
  
- **Comunicación:**
  - Múltiples rondas = latencia alta
  - Compresión de gradientes/pesos
  - Cuantización (int8, float16)
  
- **Privacidad diferencial:**
  - Ruido añadido a gradientes
  - Trade-off privacidad vs. utilidad
  - Conceptos epsilon-delta
  
- **Heterogeneidad de sistemas:**
  - Diferentes hardwares (Jetson, CPU, GPU)
  - Variabilidad de velocidad de procesamiento
  - Asincronía en actualizaciones

### 4.5 Ensembles en Federated Learning
- **Ensemble de detectores:**
  - Múltiples YOLO entrenados en diferentes datos
  - Fusión con NMS ponderado
  - Mejora de robustez
  
- **Ensemble de OCR:**
  - Múltiples modelos CRNN agregados
  - Voting (mayoritario o ponderado)
  - Temporal voting: consenso en múltiples frames
  - Reducción de errores sistemáticos

### 4.6 Comunicación Eficiente
- Serialización de modelos: pickle, protobuf
- Transferencia incremental de cambios (delta)
- Compresión: Base64, gzip
- Sincronía vs. Asincronía

---

## 5. OPTIMIZACIÓN PARA DISPOSITIVOS EDGE

### 5.1 Computación Edge
- Definición: procesamiento cerca de la fuente de datos
- Ventajas: latencia, privacidad, ancho de banda
- Desafíos: recursos limitados (CPU, RAM, almacenamiento)

### 5.2 Cuantización
- **Conceptos:**
  - Reducción de precisión: float32 → float16, int8
  - Pérdida de información vs. ganancia en velocidad
  
- **Tipos:**
  - Cuantización post-entrenamiento
  - Cuantización durante entrenamiento (QAT)
  
- **Implementación TFLite:**
  - Convertidor INT8
  - Quantization-aware training
  - Impacto en precisión y velocidad

### 5.3 Destilación de Modelos
- Concepto: modelo grande (maestro) → modelo pequeño (estudiante)
- Loss: combinación de cross-entropy + KL divergence
- Temperatura de suavidad
- Aplicación: comprimir OCR sin perder precisión

### 5.4 Pruning (Poda)
- Eliminación de pesos insignificantes
- Structured vs. unstructured pruning
- Impacto en velocidad y almacenamiento

### 5.5 TensorFlow Lite
- Formato optimizado para dispositivos móviles/edge
- Conversión desde Keras/SavedModel
- Quantization con representative dataset
- Interpretadores: Python, C++
- Benchmarking en hardware específico

### 5.6 Hardware Específico: Jetson Nano
- Especificaciones: ARM64, GPU NVIDIA Maxwell, 2GB RAM
- Restricciones: procesamiento paralelo limitado, memoria
- Perfiles de runtime:
  - jetson-stable: máxima compatibilidad, menor FPS
  - jetson-realtime: máximo throughput, riesgo de timeout
  - jetson-quality: balance intermedio
- TensorRT: optimización de modelos para Jetson

---

## 6. PIPELINE INTEGRADO: DETECCIÓN + OCR

### 6.1 Arquitectura End-to-End
- Captura de video en tiempo real
- Preprocesamiento per-frame
- Detección de placas (YOLO)
- Extracción de ROI (Region of Interest)
- OCR en ROI (CRNN+CTC)
- Post-procesamiento: filtrado, deduplicación
- Persistencia de evidencia (imagen + anotaciones)

### 6.2 Estrategias de Deduplicación
- **Cooldown temporal:** evitar captura repetida de misma placa
- **Deduplicación por firma:** hash de imagen para identidad
- **Deduplicación por OCR:** consenso en N frames antes de guardar
- **Estructura de datos:** caché con TTL (Time-To-Live)

### 6.3 Recuperación de Detecciones Perdidas
- Problema: placas pequeñas o distantes no detectadas
- Solución: modelo detector dual (detector secundario especializado)
- Estrategia "rescue": cada N detecciones, si hay silencio, fuerza intento con modelo secundario

### 6.4 Captura de Evidencia
- Almacenamiento de imágenes anotadas
- Metadatos: timestamp, confianza, OCR, modelo usado
- Formatos: JPEG, PNG, JSON de metadatos

---

## 7. DATASETS Y EVALUACIÓN

### 7.1 Dataset ALPR
- **Composición:**
  - Imágenes de placas en distintos ángulos, distancias, condiciones
  - Variabilidad: luz natural, artificial, sombra, noche
  - Tipos de placa: estándar, especiales
  
- **Split train/val/test:**
  - Estratificación por placa (evitar leakage)
  - Balance de clases (si es necesario)
  
- **Anotaciones:**
  - Bounding boxes (formato YOLO, COCO)
  - Texto de placa (OCR ground truth)

### 7.2 Métricas de Evaluación Globales
- **Para detección:** mAP, Precision, Recall por IoU
- **Para OCR:** CER (Character Error Rate), WER (Word Error Rate)
- **End-to-end:** tasa de éxito de lectura completa
- **Performance:** FPS en hardware target, latencia P95
- **Calibración:** confianza predicha vs. tasa de error real

### 7.3 Análisis de Errores
- Casos difíciles: placas pequeñas, lejanas, obstruidas
- Confusiones de caracteres comunes
- Sesgos por tipo de placa o condición de luz
- Auditoría de privacidad y fairness

---

## 8. IMPLEMENTACIÓN TÉCNICA

### 8.1 Stack Tecnológico
- **Frameworks:**
  - PyTorch: YOLO (detección)
  - TensorFlow/Keras: OCR CRNN+CTC
  - Customización: custom layers (CTC), callbacks
  
- **Librerías auxiliares:**
  - OpenCV: procesamiento de imágenes
  - NumPy: operaciones matriciales
  - Pillow: lectura/escritura de imágenes
  
- **Infraestructura:**
  - Versionado: Git, control de modelos
  - Logging: registro de entrenamiento, métricas
  - Serialización: pickle, base64, JSON

### 8.2 Generación de Código
- Exportación de modelos a Python puro
- Ventajas: portabilidad, sin dependencia de framework
- Desafío: custom layers (CTCLayer) no registradas
- Solución: regenerar arquitectura + serializar pesos con base64+pickle

### 8.3 Configuración Centralizada
- `runtime_config.py`: parámetros globales
- Perfiles: jetson-stable, jetson-quality, jetson-production, jetson-realtime
- Override dinámico: CLI arguments, env vars

### 8.4 Modularidad
- Separación: detección ↔ OCR ↔ orquestación
- Reusabilidad: funciones de preprocesamiento, callbacks
- Testability: mocks para componentes, fixtures de datos

---

## 9. RESULTADOS Y VALIDACIÓN

### 9.1 Benchmarks de Desempeño
- Precisión vs. velocidad trade-off
- Comparativa: modelo completo vs. cuantizado
- Impacto de compression en precisión
- Speedup en Jetson respecto a GPU/CPU de escritorio

### 9.2 Análisis de Convergencia Federada
- Curvas de loss/accuracy en rondas federadas
- Impacto de hetereogeneidad de datos (non-IID)
- Número de rondas requeridas vs. centralizado
- Comunicación total (bytes transferidos)

### 9.3 Casos de Uso Reales
- Escenarios: peaje, estacionamiento, control de tráfico
- Variabilidad de condiciones: día/noche, lluvia, polvo
- Robustez a adversarial perturbations

---

## 10. CONCLUSIONES Y PERSPECTIVAS FUTURAS

### 10.1 Síntesis del Proyecto
- Integración exitosa: detección + OCR + federated learning
- Adaptación a hardware edge (Jetson Nano)
- Privacidad preservada sin sacrificar precisión significativamente

### 10.2 Limitaciones Actuales
- Dependencia de calidad de anotaciones
- Sesgos regionales (tipo de placa, formato)
- Latencia de comunicación en federated learning
- Overhead de coordinación federada

### 10.3 Direcciones de Investigación Futura
- **Privacidad:**
  - Privacidad diferencial en agregación
  - Secure aggregation (criptografía)
  
- **Eficiencia:**
  - Federated learning asincrónico
  - Comunicación comprimida adaptativa
  - Personalización (federated multi-task learning)
  
- **Robustez:**
  - Defensa contra adversarial examples
  - Detección de anomalías
  - Explicabilidad (XAI)
  
- **Escalabilidad:**
  - Millones de dispositivos edge
  - Geo-distributed aggregation
  - Hierarchical federated learning

### 10.4 Regulación y Ética
- Cumplimiento GDPR: privacidad de datos vehiculares
- Transparencia en decisiones de IA
- Auditoría de sesgo y discriminación
- Responsabilidad en caso de error

---

## 11. REFERENCIAS ESTRUCTURALES (SUGERIDAS)

### Trabajos Seminales
- YOLOv3/v5 (Redmon et al., 2018+)
- CRNN (Shi et al., 2016)
- CTC (Graves et al., 2006)
- FedAvg (McMahan et al., 2016)

### Estándares y Benchmarks
- COCO Dataset (detección de objetos)
- ICDAR (OCR y reconocimiento de texto)
- Benchmarks de sistemas ALPR (RTSD, etc.)

### Hardware y Optimización
- TensorFlow Lite documentation
- TensorRT (NVIDIA Jetson)
- OpenVINO (Intel edge devices)

---

## NOTAS PARA LA REDACCIÓN

1. **Claridad:** Explicar conceptos por niveles (básico → avanzado)
2. **Contexto:** Justificar cada sección respecto al proyecto específico
3. **Matemática:** Incluir ecuaciones clave (CTC, FedAvg, IoU)
4. **Diagramas:** Arquitecturas de redes, flujos de datos, comunicación federada
5. **Balance:** Teoría suficiente sin exceso; énfasis en aplicación
6. **Ejemplos:** Casos concretos del proyecto (Jetson Nano, placas ecuatorianas, etc.)

