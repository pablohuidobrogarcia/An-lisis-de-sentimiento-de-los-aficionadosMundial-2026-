# Business Insights — World Cup 2026 Sentiment Analysis

*Documento dirigido a un público no técnico: marcas patrocinadoras, federaciones, medios de comunicación.*

---

## Resumen ejecutivo

Este proyecto analizó **200,316 comentarios en YouTube** (144,910 en inglés, 55,406 en español) durante el Mundial 2026 (11 de junio – 23 de julio de 2026) para medir cómo cambia la percepción del público hacia cada selección y qué factores explican esos cambios. Los resultados permiten tomar decisiones informadas sobre comunicación de marca, gestión de crisis y estrategia de patrocinio.

**Número clave**: el **46.3%** de toda la conversación fue negativa, frente a solo un **28.2%** positiva. El Mundial 2026, visto desde YouTube, fue un torneo emocionalmente polarizado y con sesgo negativo persistente.

---

## Hallazgos principales

### 1. El arbitraje fue el mayor generador de crisis de reputación

Los comentarios que mencionan **arbitraje, VAR o "rigged"** (8,971 comentarios) alcanzan un **74% de negatividad**, frente al 46% medio. El tópico *referee / rigged / VAR* reunió el **5.9%** de toda la conversación — casi el triple de negatividad que la media del torneo.

**Implicación para marcas**: Cualquier partido con polémica arbitral dispara el sentimiento negativo de forma inmediata e independiente del resultado. Evitar activar campañas de marca en esos días: el mensaje será invisible o asociado negativamente.

### 2. Las victorias no siempre mejoran el sentimiento... pero la eliminación y la derrota en la final sí lo empeoran

El sentimiento cayó de forma constante según avanzaba el torneo:

| Fase | Sentimiento positivo | Sentimiento negativo |
|------|---------------------|---------------------|
| Fase de grupos | **32.3%** (pico de optimismo) | 39.3% |
| Octavos de final | 25.3% | **52.1%** (el peor momento) |
| Cuartos de final | 22.8% | 48.8% |
| Semifinales | 27.4% | 45.5% |
| 3er puesto | 26.2% | 47.3% |
| Final | 29.6% | 49.5% |

**Implicación**: La fase de grupos fue el momento de mayor expectación positiva. En eliminatorias la conversación se vuelve negativa de forma estructural. Las marcas deberían concentrar su inversión publicitaria optimista en la fase de grupos y pasar a tono de gestión/escucha en octavos.

### 3. El tema que más impacta en el sentimiento: la actuación de las superestrellas

Una única narrativa dominó todo el torneo: **Argentina y Messi**. El 66% de los comentarios cayó en el tópico *argentina / messi / team*. Los jugadores más mencionados:

| Jugador | Menciones |
|---------|-----------|
| Lionel Messi | 13,004 |
| Kylian Mbappé | 3,229 |
| Harry Kane | 1,605 |
| Lamine Yamal | 1,484 |
| Neymar | 1,318 |
| Jude Bellingham | 779 |

Messi concentró **4× más menciones** que Mbappé y **8× más** que Kane. En las selecciones con superestrellas (Argentina, Francia, Inglaterra), el sentimiento depende de la actuación individual más que del rendimiento colectivo.

**Implicación**: En mercados como Argentina, las campañas deben girar en torno a figuras individuales, no al equipo como colectivo. Para España, la ausencia de una estrella dominante (Yamal es lo más cercano) implica un sentimiento más plano y menos volátil.

### 4. Ventana de oportunidad: el pico de conversación tras la final

El día posterior a la final (20 de julio) fue el pico absoluto de volumen con **15,071 comentarios** y un **58.8% de negatividad**. El 8 de julio alcanzó el máximo de negatividad diaria (61.8%).

**Implicación**: Los momentos de máximo volumen son también los de máxima negatividad. No es buen momento para mensajes institucionales; sí para escucha activa y monitorización en tiempo real.

### 5. Comparativa entre selecciones

| Selección | POS | NEG | NEU | Volatilidad | Tema dominante |
|-----------|-----|-----|-----|-------------|----------------|
| Brasil | 30.3% | 40.7% | 29.0% | Media | Neymar / Vinícius |
| Inglaterra | 28.5% | 45.5% | 26.0% | Media | Kane / Bellingham |
| Argentina | 27.8% | 50.5% | 21.8% | Muy alta | Messi |
| Francia | 26.6% | 45.8% | 27.7% | Alta | Mbappé |
| España | 26.3% | 49.5% | 24.2% | Media | Lamine Yamal |

**Implicación**: **Brasil** fue la selección mejor percibida (30.3% POS) y **Argentina** la peor percibida en términos de negatividad (50.5% NEG), pese a dominar la conversación. Las marcas asociadas a Brasil disfrutaron del mejor contexto reputacional del torneo.

---

## Recomendaciones para marcas patrocinadoras

1. **Activar campañas optimistas en la fase de grupos**: es el único período con sentimiento positivo superior al 30%.
2. **Evitar días de alta polémica arbitral**: un partido con controversia de VAR genera hasta 74% de mensajes negativos que se engullen cualquier mensaje de marca.
3. **Personalizar por mercado**: en Argentina y Francia el sentimiento se ancla a estrellas individuales (Messi, Mbappé); en España y Brasil, al equipo.
4. **Medir antes de actuar**: los cambios de sentimiento deben monitorizarse por fase y tema, no solo por resultado, porque victorias y derrotas explican solo parte de la variación.

---

## Metodología

- **Fuente**: YouTube (canales FIFA, ESPN, FOX Soccer, TUDN, beIN SPORTS).
- **Idiomas**: Español e inglés (72% EN / 28% ES).
- **Modelo de sentimiento**: cardiffnlp XLM-RoBERTa fine-tuned sobre etiquetas manuales, con **72.6% de accuracy en test** (F1 macro 0.71). Se partió de pysentimiento (ES) y RoBERTa (EN).
- **Topic modeling**: BERTopic con embeddings multilingües (20 temas tras consolidación e interpretación de outliers).
- **NER**: spaCy + diccionario propio de jugadores del Mundial 2026 (24,794 comentarios con menciones de jugadores).
- **Período**: 11 de junio – 23 de julio de 2026 (grupos, eliminatorias y final).

---

## Limitaciones

- Los datos provienen exclusivamente de YouTube, que tiene un sesgo hacia usuarios que buscan activamente contenido deportivo.
- Solo se analizan comentarios en español e inglés.
- El modelo puede fallar en casos de sarcasmo o ironía compleja.
- Los hallazgos son **correlacionales**, no causales: el análisis pre/post partido se retiró del dashboard final por la falta de fiabilidad de las ventanas temporales.
- La detección de marcas patrocinadoras (NER de brands) no produjo resultados: las conclusiones de marca se infieren de los temas, no de menciones explícitas.

---

*Documento generado como parte del proyecto portfolio de Pablo Huidobro García.*
*Última actualización: agosto 2026.*
