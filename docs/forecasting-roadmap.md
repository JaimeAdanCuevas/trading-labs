# Plan para una herramienta de pronóstico de mercado

**Estado:** diseño inicial  
**Actualizado:** 2026-11-18

## Objetivo

Estimar la distribución probable del rendimiento de un activo en horizontes definidos y comunicar tanto la incertidumbre como las condiciones que podrían invalidar la señal. El sistema no debe afirmar que conoce el próximo precio exacto.

Primera familia de etiquetas:

- **Sube:** rendimiento futuro supera un umbral positivo después de costes estimados.
- **Baja:** rendimiento futuro cae por debajo de un umbral negativo después de costes.
- **Neutral:** movimiento dentro de la banda intermedia.

Los umbrales deben depender del activo, la volatilidad y el horizonte. La salida debe incluir probabilidades para las tres etiquetas, rango de rendimiento plausible, fecha de cálculo, horizonte y principales factores. Si la evidencia es débil, debe poder abstenerse.

## Horizontes y universo inicial

- Fijar **5 sesiones hábiles como horizonte principal** del primer predictor.
- Evaluar **1 y 20 sesiones como horizontes secundarios** para comparar ruido, estabilidad y utilidad; no sustituir el objetivo principal después de ver resultados.
- Mantener aparte un análisis fundamental de **12 a 24 meses**; no mezclarlo con el pronóstico táctico. La tesis de inversión de BABA a dos años sirve como caso de estudio, no como validación del modelo.
- Comenzar con BABA, SPY y QQQ; añadir FXI, KWEB y 9988.HK sólo cuando se resuelvan horarios, tipo de cambio, dividendos, ADR y equivalencia entre listados.

## Datos y trazabilidad

1. Precios ajustados, volumen, dividendos, splits, volatilidad y calendario de mercado.
2. Fundamentos y resultados con su fecha de publicación real, no sólo el trimestre al que pertenecen.
3. Tasas, moneda, índices comparables y variables macro relevantes.
4. Noticias o sentimiento sólo cuando tengamos procedencia, hora de publicación, licencia y una regla para información revisada o eliminada.
5. Cada observación debe distinguir `fecha_del_evento` de `fecha_disponible`; conservar versión de fuente, símbolo y unidad.

La primera etapa debe preferir datos pequeños y auditables. Evitamos incorporar de entrada cientos de variables que faciliten sobreajuste.

## Validación que puede descubrir fallas

- Dividir por tiempo: entrenar en el pasado y evaluar en periodos posteriores mediante ventanas walk-forward. No barajar días al azar.
- Simular que la decisión se toma después de que la información estuvo disponible. Los resultados publicados después del cierre no pueden predecir ese mismo cierre.
- Comparar contra referencias simples: rendimiento cero, comprar y mantener, tendencia/momentum y el cruce de medias ya explorado en `y_finance.py`.
- Reservar un periodo final intacto para la evaluación; registrar cada prueba para reducir la selección accidental del mejor resultado.
- Incluir comisión, spread, deslizamiento, conversión de moneda, dividendos y, cuando corresponda, disponibilidad/costo de préstamo.
- Reportar Brier score, log loss y calibración de probabilidades; además de rendimiento neto, drawdown, rotación, exposición y estabilidad por régimen. La exactitud direccional sola no basta.
- Medir los resultados por activo y horizonte. Una señal que sólo funciona en un tramo, un activo o antes de costes no cuenta como ventaja estable.

## Orden de trabajo

1. **Auditoría del prototipo y de fuentes:** documentar errores de disponibilidad temporal y limitaciones de los datos actuales.
2. **Base reproducible:** esquema de datos, fuentes, unidades, fechas de publicación y snapshots históricos.
3. **Referencias:** generar predicciones constantes y estrategias sencillas; registrar sus métricas.
4. **Primer modelo interpretable:** calibrar probabilidades con pocos predictores y walk-forward.
5. **Prueba de robustez:** costes, periodos bajistas/laterales/alcistas, cambios de parámetros y periodo final intacto.
6. **Seguimiento en papel:** producir predicciones fechadas sin operar y compararlas con los resultados posteriores.
7. **Interfaz:** sólo después de demostrar calibración y estabilidad, mostrar escenarios, razones, incertidumbre y límites.

## Criterio para decir que sirve

El modelo debe mejorar referencias sencillas fuera de muestra en más de una ventana y mantener probabilidades razonablemente calibradas después de costes. Si no lo hace, el resultado válido de la investigación es que aún no encontramos una ventaja predictiva.
