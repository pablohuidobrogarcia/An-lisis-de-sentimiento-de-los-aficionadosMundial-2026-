# Instrucciones para el Etiquetado Manual de Sentimiento

## Objetivo

Asignar una etiqueta de sentimiento (POS / NEU / NEG) a cada comentario para
evaluar la precisión de los modelos automáticos de análisis de sentimiento.

## Definiciones

| Etiqueta | Significado | Ejemplos en contexto futbolístico |
|----------|-------------|-----------------------------------|
| **POS** (Positive) | El comentario expresa alegría, orgullo, esperanza, admiración o celebración. | *"¡Vamos México! Gran partido"*, *"Qué golazo"*, *"Orgulloso de esta selección"* |
| **NEU** (Neutral) | El comentario es descriptivo, factual, una pregunta, o no expresa una emoción clara. | *"El partido es a las 3"*, *"México vs Sudáfrica"*, *"Alguien sabe quién anotó?"* |
| **NEG** (Negative) | El comentario expresa frustración, decepción, enojo, crítica o tristeza. | *"Qué vergüenza de arbitraje"*, *"Jugaron pésimo"*, *"Nos eliminaron otra vez"* |

## Casos difíciles

- **Sarcasmo / ironía**: Si el comentario es sarcástico, usa la etiqueta que
  corresponda a la intención real (normalmente NEG si critica aunque use
  palabras positivas). Anótalo en la columna `notas` si quieres.
- **Sentimiento mixto**: Si el comentario contiene elementos POS y NEG,
  elige la emoción dominante. Si es imposible decidir, usa NEU.
- **Comentarios sobre el video vs. el partido**: Un comentario como
  *"Buen video, mal resumen"* habla del video (POS) y del contenido (NEG).
  Evalúa el sentimiento global o la emoción principal hacia el fútbol/equipo.
- **Solo emojis**: Si el comentario contiene solo emojis (ej: "🇲🇽🔥"),
  etiqueta según la emoción que transmiten. Si no es claro, usa NEU.
- **Menciones a otros usuarios**: Ignora las menciones (@usuario) y
  concéntrate en el contenido del mensaje.
- **Idioma**: Etiqueta basándote en el contenido, aunque tenga palabras
  mezcladas. La columna `language` es solo de referencia.

## Instrucciones

1. Abre el archivo `manual_labels_template.csv` en Excel, Google Sheets o
   cualquier editor CSV.
2. Para cada fila, escribe en la columna `manual_label` una de las tres
   etiquetas: `POS`, `NEU` o `NEG` (sin comillas, mayúsculas).
3. Si un comentario no se puede etiquetar (texto sin sentido, spam que pasó
   el filtro), déjalo vacío — será excluido de la evaluación.
4. Guarda el archivo y ejecuta:
   ```
   python evaluation/evaluate_models.py --evaluate
   ```

## ¿Por qué esto es importante?

El etiquetado manual es la única forma de medir la precisión real del modelo.
Sin esta evaluación, no sabríamos si el modelo está funcionando bien o si está
cometiendo errores sistemáticos (por ejemplo, clasificando todo como POS por
el sesgo de los comentarios positivos en el dataset de entrenamiento). Los
resultados de esta evaluación se incluirán en el TFG como la métrica de
calidad del pipeline de sentimiento.
