# Dataset

Este repositorio usa dos fuentes principales:

- `dataset_alpr/`
- `dataset_publico_placas/`

También existe una variante `dataset_publico_placas_smallfar/` para casos de cámara alejada.

Consulta también [origen_licencia_dataset.md](origen_licencia_dataset.md) para dejar formalmente escrita la trazabilidad.

## Regla de partición train/val.

Se sigue una regla de partición del dataset 60/30/10:

60%: Entrenamiento
30%: Validación
10%: Pruebas

## Número de imágenes por conjunto.

Entrenamiento: 11428 imágenes
Validación: 5730 imágenes
Pruebas: 1890 imágenes
