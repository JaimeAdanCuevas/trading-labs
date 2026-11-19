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

## Familias de variables: sumar evidencia, no acumular indicadores

Vamos a considerar las variables que puedan aportar información para cinco sesiones, pero añadiremos cada familia por separado y la conservaremos sólo si mejora el resultado fuera de muestra:

1. **Noticias y eventos:** hora real de primera publicación, tipo de noticia, novedad, sorpresa contra el consenso disponible y reacción inicial. La polaridad del texto por sí sola no basta; una noticia “buena” puede ser peor de lo esperado o ya estar descontada.
2. **Técnico y precio:** retornos rezagados, gaps, tendencia, reversión, volatilidad realizada, rango, volumen y amplitud. Los indicadores deben calcularse sólo con datos conocidos al momento de la señal.
3. **Fundamentales:** resultados, guía, revisiones, márgenes, caja y valoración. Para cinco sesiones son especialmente útiles como catalizadores o contexto; para 12–24 meses su papel puede ser distinto.
4. **Macro y régimen:** índices, tasas, curva, FX, commodities, volatilidad y correlaciones; en BABA, sumar proxies chinos con horario y publicación correcta.
5. **Flujos y microestructura:** spread, profundidad, desequilibrio de órdenes, volatilidad, posiciones y opciones sólo cuando tengamos datos con hora, calidad y cobertura adecuadas. Volumen no es sinónimo de presión compradora neta.
6. **Anomalías e integridad de mercado:** cancelaciones de órdenes, órdenes no genuinas, operaciones relacionadas y posibles patrones de spoofing/layering requieren datos de órdenes y trazas por cuenta/venue. Con precios y volumen diarios sólo podemos marcar “movimiento anómalo”, nunca concluir manipulación.
7. **Financiamiento y conexiones entre empresas:** vencimientos, compromisos, deuda, inversión de proveedores y ventas a clientes vinculados pueden afectar riesgo sistémico; estos hechos deben verificarse compañía por compañía y no inferirse sólo de un gráfico.

La regla de selección es una prueba de ablación: comparar baseline, baseline más una sola familia y combinaciones justificadas. Registrar todas las variantes. “Más variables” no significa “más información”; puede significar más ruido y más oportunidades para sobreajustar.

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
