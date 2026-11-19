# Cinco sesiones: hipótesis sobre cómo se mueve el mercado

- **Propósito:** convertir intuiciones sobre precio, noticias, liquidez y posicionamiento en hipótesis que puedan rechazarse con datos.
- **Horizonte inicial:** cinco sesiones hábiles.
- **Estado:** guía de investigación; no es una explicación completa del mercado ni una estrategia validada.

## Primera corrección de enfoque

No hay un único “motor” del mercado. En un horizonte de cinco sesiones, la variación de precio puede mezclar información nueva, demanda urgente, liquidez disponible, cobertura de posiciones y ruido. El mismo dato puede producir movimientos diferentes cuando cambia el posicionamiento, la profundidad del libro o el régimen de volatilidad.

Por eso separaremos tres preguntas:

1. **Qué cambió en el valor esperado?** Resultados, pronósticos, tasas, moneda, regulación o noticias.
2. **Quién debe comprar o vender ahora?** Flujos, rebalanceos, coberturas, rescates, margen y reglas de riesgo.
3. **Cuánto mueve el precio ese flujo?** Depende de liquidez y profundidad; una orden neta similar puede tener distinto impacto en días distintos.

El precio observado es resultado conjunto de estas fuerzas. Una historia fundamental coherente no basta para anticipar el signo de cinco días. Tampoco todo movimiento de corto plazo significa que cambió el valor fundamental.

## Qué dice la evidencia y qué no dice

- En datos de libro de órdenes de 50 acciones de EE. UU., Cont, Kukanov y Stoikov encuentran que, en intervalos cortos, el desequilibrio de órdenes de compra/venta se relaciona con cambios de precio, y que el impacto aumenta cuando la profundidad baja. Su muestra y escala son de alta frecuencia: esto justifica medir liquidez y flujo si los datos están disponibles; **no demuestra que una medida diaria prediga el rendimiento de cinco sesiones**. [Artículo](https://arxiv.org/abs/1011.6402)
- Heston, Korajczyk y Sadka relacionan ciertos patrones de reversión de muy corto plazo con desequilibrios transitorios de liquidez y el rebote bid-ask. Las fricciones de ejecución pueden borrar una señal aparente. [Artículo](https://doi.org/10.1111/j.1540-6261.2010.01573.x)
- Moskowitz, Ooi y Pedersen encuentran momentum de series de tiempo en horizontes de uno a doce meses para una cesta de futuros líquidos; no es evidencia directa de que perseguir retornos de cinco días funcione. [Artículo](https://doi.org/10.1016/j.jfineco.2011.11.003)
- Bailey, Borwein, López de Prado y Zhu formalizan el riesgo de elegir la mejor estrategia después de probar muchas. Cuantas más variantes probemos, menos informativo será el mejor resultado dentro de muestra. [Artículo](https://escholarship.org/uc/item/4w1110bb)

Conclusión de diseño: los mecanismos aportan hipótesis para probar, no permiso para declarar una ventaja.

## Qué variables vamos a observar

| Familia | Variables candidatas | Tratamiento cuidadoso |
|---|---|---|
| Noticias | Hora de primera difusión, fuente, tipo de evento, novedad, sorpresa frente al consenso, reacción inicial y revisiones. | Usar el dato que el mercado podía leer entonces. “Sentimiento positivo” no equivale a sorpresa positiva; no asignar hora de un agregador como si fuera hora de publicación. |
| Técnica | Rendimientos 1/5/20 sesiones, gaps, tendencia, reversión, rango, volumen relativo, volatilidad realizada, drawdown y amplitud. | Son transformaciones de precios pasados, no evidencia independiente por defecto. Evitar calcularlos usando el cierre de entrada. |
| Fundamental | Ventas, márgenes, flujo de caja, guía, estimaciones, revisiones y valoración. | Vincular a la hora en que se publicaron. Para cinco sesiones suelen pesar más alrededor de resultados/eventos; para dos años pueden cambiar la tesis y el valor esperado. |
| Macro y régimen | Retornos de índices/sectores, tasas y cambios de curva, FX, commodities, volatilidad, correlación y calendario de anuncios. | Definir estados con información disponible en `t`; probar si la relación cambia entre estados. |
| Liquidez y posicionamiento | Spread, profundidad, desequilibrio bid/ask, volumen negociado, opciones, interés corto y flujos de ETF si hay historia fiable. | El volumen por sí solo no dice qué lado fue más agresivo. Ordenes/cotizaciones intradía pueden ser necesarias; revisar costos y cobertura por mercado. |
| Financiamiento/reflexividad | Deuda, vencimientos, compromisos de compra/arrendamiento, financiamiento a clientes, inversiones cruzadas y dependencia de pocas contrapartes. | Confirmar en reportes; diferenciar acuerdos anunciados, obligaciones firmes y gasto ya desembolsado. Una relación comercial no implica ingreso ficticio. |
| Integridad de mercado | Órdenes canceladas repetidamente, patrones en varios niveles, wash trades, movimiento anormal en cierre y actividad coordinada entre instrumentos. | Sólo datos diarios permiten una alerta estadística de anomalía. No permiten atribuir intención ni acusar manipulación. Hace falta revisar órdenes y operaciones por cuenta, tiempo, venue e instrumentos relacionados. |

La SEC describe el spoofing/layering como órdenes no genuinas usadas para crear una apariencia falsa de interés y mover precios; FINRA indica que las alertas se deben adaptar al tipo de producto, flujo y actividad relacionada. Esas definiciones muestran por qué una vela grande o un pico de volumen no bastan para declarar manipulación. [SEC: caso de spoofing/layering](https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-10989-s), [FINRA: vigilancia de trading manipulativo](https://www.finra.org/sites/default/files/2024-01/2024-annual-regulatory-oversight-report.pdf).

Para eventos corporativos, guardar tanto la hora de publicación como la hora de presentación al regulador; usar la primera difusión pública verificable. [SEC: guía sobre reportes 8-K](https://www.sec.gov/investor/pubs/readan8k.pdf).

## Hipótesis refutables para cinco sesiones

| Hipótesis | Señal observable | Cómo podría fallar |
|---|---|---|
| El movimiento reciente contiene persistencia sólo en algunos regímenes. | Retornos rezagados, tendencia de varias semanas, volatilidad y régimen de mercado. | Costos, reversión posterior, cambio de régimen o señal concentrada en pocos episodios. |
| Una caída con poca profundidad tiene una distribución de pérdidas más asimétrica. | Spread, volumen monetario, volatilidad, gaps y, si existen, profundidad/desequilibrio del libro. | Volumen diario no equivale a profundidad; una caída puede incorporar información permanente y no revertir. |
| Noticias importan por su sorpresa frente a expectativas, no por ser “buenas” o “malas” en abstracto. | Hora de publicación, consenso previo, magnitud de sorpresa y reacción inicial. | Consenso sesgado, cifra revisada, noticia anticipada o evento simultáneo. |
| El entorno de mercado cambia la señal de un activo. | Rendimiento del índice, amplitud, correlación, volatilidad, tasas, moneda y sector. | Clasificación de régimen retrospectiva, inestable o definida con información futura. |
| Apalancamiento y operaciones abarrotadas pueden amplificar una caída. | Volatilidad, gaps, volumen anormal, spreads y proxies de posicionamiento/financiamiento. | Las posiciones reales no siempre son observables; proxy equivocado crea una narrativa falsa. |

## Contrato de la primera predicción

- **Momento de cálculo:** después del cierre de la sesión `t`, con datos publicados hasta esa hora.
- **Entrada simulada:** apertura de la sesión `t+1`, para no fingir ejecución al cierre usado para calcular la señal.
- **Horizonte:** mantener cinco sesiones; salida al cierre de `t+5`.
- **Resultado:** rendimiento total de ese intervalo, incluyendo dividendos cuando corresponda y costos de operación razonables.
- **Clasificación:** `sube`, `neutral` o `baja`, con umbral neutral elegido de antemano según costos y volatilidad. Publicar también la probabilidad y el rango, no sólo la clase.
- **Comparación:** probabilidad histórica base, comprar y mantener cuando aplique, y reglas sencillas de tendencia/reversión sin ajuste de parámetros posterior al test.

El cierre de BABA en EE. UU. y el de 9988.HK pertenecen a horarios distintos. No se puede sincronizar ambos con el mismo “día” sin definir qué información ya estaba disponible.

## El simulador que acompaña esta guía

`simulator/index.html` deja variar un cambio hipotético de valor fundamental, sorpresa noticiosa, tendencia técnica previa, flujo unilateral anómalo, régimen de volatilidad, profundidad, estrategias de tendencia y retroalimentación de ventas forzadas. Dibuja dos distribuciones con los mismos choques aleatorios: una con el bucle de ventas y contracción de liquidez activo y otra sin ese bucle.

Es un modelo de juguete para preguntar “¿qué supuesto vuelve frágil el resultado?”. Sus parámetros no fueron estimados con datos históricos y sus probabilidades no son pronósticos de BABA, SPY ni QQQ. El modelo real tendrá que estimar parámetros con snapshots históricos y demostrar mejora fuera de muestra.

## Orden de investigación

1. Congelar este contrato del objetivo y generar la etiqueta de cinco sesiones sin sesgo temporal.
2. Medir baselines por activo y período; incluir el efecto de gaps y costos.
3. Añadir variables por familia de una en una: precio/volatilidad, mercado, eventos/fundamentos y liquidez.
4. Evaluar con ventanas walk-forward, periodo final no tocado, calibración probabilística y estabilidad por subperíodo.
5. Registrar todas las variantes que se probaron; nunca informar sólo el mejor backtest.
6. Ejecutar señales en papel y comparar ejecución/derrape realista antes de considerar una estrategia operable.

## Fuentes adicionales que debemos conseguir

Para pasar del simulador educativo a investigación histórica necesitaremos calendario de publicación de noticias/resultados con hora; precio bid/ask o spread; volumen monetario y, si es posible, datos de libro; benchmark y sectores; tasas/FX; disponibilidad histórica sin revisiones; splits/dividendos y costos para ADR/listado de Hong Kong. `yfinance` puede servir para exploración, pero no garantiza por sí solo los datos point-in-time ni el detalle de microestructura que exige esta hipótesis.
