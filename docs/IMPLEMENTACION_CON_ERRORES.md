# Implementacion Con Errores y Troubleshooting

## Objetivo del documento
Registrar problemas reales encontrados durante el proyecto ALPR (PC + Jetson Nano J1010), su causa raiz, la solucion aplicada y el estado final.

## Resumen ejecutivo
- Se presentaron bloqueos en instalacion, camara CSI monocroma, dependencias de OCR/TF y rendimiento en edge.
- El pipeline se estabilizo en Jetson con perfil realtime y mejoras de OCR.
- Se descarto TensorRT por restriccion de almacenamiento en esta etapa.

## Matriz de errores

### 1) TensorFlow bloqueaba el arranque
- Sintoma: el script fallaba al iniciar por import de TensorFlow en entornos donde no estaba instalado.
- Causa raiz: import obligatorio en tiempo de carga.
- Solucion: TensorFlow opcional con carga diferida y validacion solo cuando se necesita.
- Estado: resuelto.

### 2) Camara CSI mostraba frames negros
- Sintoma: la camara abria pero se veia negro en OpenCV.
- Causa raiz: formato monocromo Y16/GREY sin normalizacion adecuada.
- Solucion: normalizacion por percentiles, CLAHE y conversion segura a BGR uint8.
- Estado: resuelto.

### 3) Reconexiones y congelamientos de stream
- Sintoma: stream se quedaba sin frames nuevos.
- Causa raiz: lectura bloqueante y perdida de frame en backend.
- Solucion: lector de camara en hilo, watchdog de freeze y reconexion automatica.
- Estado: resuelto.

### 4) OCR inestable (saltos de texto)
- Sintoma: placas con texto variable entre frames.
- Causa raiz: lecturas de baja confianza y ruido por frame.
- Solucion: voto temporal OCR, umbral de aceptacion y estado HOLD antes de publicar.
- Estado: resuelto.

### 5) Deteccion de placas pequenas insuficiente
- Sintoma: placas lejanas/small no siempre detectadas.
- Causa raiz: costo de deteccion alto para rescate continuo.
- Solucion: rescate de placas pequenas periodico y controlado (solo cada N ciclos y cuando no hay deteccion viva).
- Estado: resuelto con tuning conservador.

### 6) TensorRT/trtexec no disponible por espacio
- Sintoma: instalacion incompleta de paquetes y bucle de apt por falta de almacenamiento.
- Causa raiz: Jetson Nano con espacio insuficiente para stack completo TensorRT.
- Solucion: purga de paquetes rotos, limpieza de cache, abandono temporal de rama TRT.
- Estado: mitigado (pipeline principal operativo sin TRT).

### 7) Dependencias OCR no resueltas en entorno local de edicion
- Sintoma: warnings de import (paddleocr/rapidocr/tensorflow) en analisis estatico.
- Causa raiz: entorno local diferente al runtime objetivo.
- Solucion: mantener flujo tolerante y documentar entorno de ejecucion real.
- Estado: aceptado.

## Lecciones aprendidas
- En edge, estabilidad > complejidad: primero pipeline robusto, luego aceleracion avanzada.
- En camaras industriales monocromas, la normalizacion de entrada es critica.
- El OCR en vivo mejora mucho con consenso temporal y control de frecuencia.
- La observabilidad (logs, snapshots, capturas) evita diagnosticos a ciegas.

## Criterios de salida de errores (cumplidos)
- Pipeline en vivo estable sin caidas.
- Deteccion + OCR con captura automatica de evidencia.
- Rendimiento objetivo minimo sostenido alrededor de 5 FPS en perfil realtime.
- Ruta clara para retomar TensorRT cuando haya espacio suficiente.
