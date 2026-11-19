# Trading Labs

Laboratorio para investigar mercados y construir, paso a paso, herramientas que estimen movimientos de precios con probabilidades, datos reproducibles y validación fuera de muestra.

El repo hoy contiene un prototipo inicial de análisis histórico con `yfinance.py`. Las investigaciones y decisiones de metodología se documentan antes de convertirlas en señales o código de producción.

## Empezar aquí

- [Plan de la herramienta de pronóstico](docs/forecasting-roadmap.md): objetivo, horizontes, datos, validación y criterios para aceptar un modelo.
- [Cómo estudiar el movimiento a cinco sesiones](docs/market-mechanics.md): mapa de hipótesis y contrato de predicción.
- [Simulador de mecanismos de mercado](simulator/index.html): laboratorio de escenarios con liquidez, volatilidad, posicionamiento y ventas forzadas.
- [Investigación de Alibaba y el ciclo de IA](research/2026-11-18-alibaba-ai-market.md): ficha con corte al 18 de noviembre de 2026, incluyendo comercio, cloud, T-Head, capacidad de centros de datos y riesgos de financiación circular.
- [Prototipo existente](y_finance.py): descarga datos históricos y explora cruces de medias móviles.

## Reglas del laboratorio

1. Una salida del modelo es una probabilidad condicionada a un horizonte, no una certeza ni un precio objetivo garantizado.
2. Cada investigación guarda su fecha de corte, fuentes y qué datos aún no estaban disponibles en esa fecha.
3. Comparamos cualquier modelo con referencias sencillas y medimos su resultado neto de costes, con periodos de validación cronológicos.
4. Separamos hechos publicados, inferencias y escenarios. Actualizamos las tesis cuando aparecen datos que las contradicen.
5. No guardamos aquí datos personales de cartera, costos de entrada ni tamaños de posición. El repo es público.

## Estado

La primera etapa es de investigación y diseño. El simulador es un modelo de mecanismos sin calibración histórica; no hay todavía un modelo predictivo validado ni señales listas para operar.

## Uso y pruebas del simulador

Abre `simulator/index.html` en un navegador. Para ejecutar sus pruebas de reproducibilidad, sensibilidad y límites con Node.js: `node --test tests/market-sim.test.mjs`.
