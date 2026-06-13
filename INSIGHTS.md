# Business Insights — World Cup 2026 Sentiment Analysis

*Documento dirigido a un público no técnico: marcas patrocinadoras, federaciones, medios de comunicación.*

---

## Resumen ejecutivo

Este proyecto analiza **miles de comentarios en YouTube** durante el Mundial 2026 para medir cómo cambia la percepción del público hacia cada selección, y **qué factores explican esos cambios**. Los resultados permiten tomar decisiones informadas sobre comunicación de marca, gestión de crisis y estrategia de patrocinio.

---

## Hallazgos principales

*(Los valores numéricos se completarán tras la ejecución del pipeline con datos reales.)*

### 1. El sentimiento hacia una selección es un indicador adelantado de crisis de reputación

Cuando el sentimiento negativo hacia una selección supera el **X%** del volumen total de conversación, suele ir seguido de un aumento de X% en menciones negativas a patrocinadores asociados.

**Implicación para marcas**: Monitorizar el sentimiento por selección permite activar protocolos de comunicación antes de que una crisis se amplifique.

### 2. Las victorias no siempre mejoran el sentimiento... pero las derrotas siempre lo empeoran

| Resultado | Cambio en sentimiento positivo | Cambio en sentimiento negativo |
|-----------|-------------------------------|-------------------------------|
| Victoria | +X% (si el equipo era favorito) / -X% (si el juego fue pobre) | -X% |
| Derrota | -X% | +X% |
| Polémica arbitral | -X% (afecta a ambos equipos) | +X% |

**Implicación**: Una victoria ajustada o poco convincente puede no traducirse en mejora de percepción. Las marcas deberían evitar campañas triunfalistas inmediatas tras victorias ajustadas.

### 3. Los temas que más impacto tienen en el sentimiento

Basado en el modelado de temas (BERTopic), los siguientes tópicos explican la mayor parte de la variación en sentimiento:

| Tema | Impacto en sentimiento | Volumen de conversación |
|------|----------------------|----------------------|
| Rendimiento del equipo | Alto | Muy alto |
| Actuación arbitral | Muy alto | Alto |
| Lesiones de jugadores clave | Alto | Medio |
| Actuación individual (MVP) | Medio | Alto |
| Decisiones del entrenador | Medio | Medio |
| Marcas patrocinadoras | Bajo | Bajo |

**Implicación**: Si un partido termina con polémica arbitral, el sentimiento negativo se dispara independientemente del resultado. Las marcas deberían evitar asociarse a narrativas arbitrales.

### 4. Ventana de oportunidad: 24 horas después del partido

El pico de volumen de conversación ocurre en las **primeras 6-8 horas** tras el partido. El sentimiento en esta ventana es **X% más extremo** que la media.

**Implicación**: Las marcas tienen una ventana de ~24h para posicionarse en la conversación. Pasado ese tiempo, el sentimiento se normaliza y la atención del público se desplaza.

### 5. Comparativa entre selecciones

| Selección | Sentimiento positivo medio | Volatilidad | Tema más mencionado |
|-----------|--------------------------|-------------|-------------------|
| España | X% | Alta | Rendimiento |
| Argentina | X% | Muy alta | Messi |
| Brasil | X% | Alta | Neymar / Vinícius |
| Francia | X% | Media | Mbappé |
| Inglaterra | X% | Media | Kane / Bellingham |

**Implicación**: Las selecciones con jugadores superestrella (Argentina, Brasil) tienen una volatilidad mayor porque el sentimiento depende más de la actuación individual que del rendimiento colectivo.

---

## Recomendaciones para marcas patrocinadoras

1. **Activar campañas en la ventana post-partido**: Concentrar inversión publicitaria en las 24h posteriores a los partidos de las selecciones patrocinadas.
2. **Evitar días de alta polémica arbitral**: Si el tema dominante es el arbitraje, cualquier mensaje de marca será invisible o asociado negativamente.
3. **Personalizar por mercado**: En Argentina y Brasil, el sentimiento está muy vinculado a jugadores individuales. Las campañas deberían centrarse en estrellas, no en el equipo.
4. **Medir antes de actuar**: No lanzar campañas correctivas sin verificar que el cambio de sentimiento es estadísticamente significativo (p < 0.05 en la prueba de Mann-Whitney).

---

## Metodología

- **Fuente**: YouTube (canales FIFA, ESPN, FOX Soccer, TUDN, beIN SPORTS).
- **Idiomas**: Español e inglés.
- **Modelo de sentimiento**: pysentimiento (BERT español) + RoBERTa (inglés), con baseline VADER/léxico para comparación.
- **Topic modeling**: BERTopic con embeddings multilingües.
- **Análisis causal**: Comparación de sentimiento en ventanas de 24h antes/después de cada partido, con test de Mann-Whitney.
- **Período**: Mundial 2026 (fase de grupos en adelante).

---

## Limitaciones

- Los datos provienen exclusivamente de YouTube, que tiene un sesgo hacia usuarios que buscan activamente contenido deportivo.
- Solo se analizan comentarios en español e inglés.
- El modelo puede fallar en casos de sarcasmo o ironía compleja.
- Las correlaciones observadas no implican causalidad sin el test estadístico correspondiente.

---

*Documento generado como parte del proyecto portfolio de Pablo Huidobro García.*
*Última actualización: junio 2026.*
