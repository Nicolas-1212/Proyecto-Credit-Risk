# Proyecto-Credit-Risk

## Integrantes
- Nicolás Ramírez Rodríguez (Integrante A)
- Samuel Arteaga (Integrante B)

## Fuente de los datos
[Credit Risk Dataset en Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
El conjunto de datos contiene información financiera y demográfica de diversas personas que han solicitado préstamos. Incluye datos sobre sus ingresos, situación de vivienda, propósito del préstamo y su historial crediticio, con el objetivo principal de predecir o analizar la probabilidad de incumplimiento de pago (default).

## Estructura de los datos
El conjunto de datos original cuenta con *32,581 registros* y *12 variables*. 
Cada fila representa *una solicitud de préstamo* realizada por un individuo, detallando su perfil personal y las características del crédito solicitado. 

## Diccionario de datos
| Variable | Tipo | Descripción | Valores / unidades |
|---|---|---|---|
| person_age | Numérica discreta | Edad del solicitante | Años |
| person_income | Numérica discreta | Ingreso anual del solicitante | Dólares |
| person_home_ownership | Categórica | Situación de vivienda actual | RENT, OWN, MORTGAGE, OTHER |
| person_emp_length | Numérica continua | Tiempo de antigüedad laboral | Años |
| loan_intent | Categórica | Propósito o motivo del préstamo | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| loan_grade | Categórica | Calificación de riesgo del préstamo | A, B, C, D, E, F, G |
| loan_amnt | Numérica discreta | Monto total del préstamo solicitado | Dólares |
| loan_int_rate | Numérica continua | Tasa de interés asignada al préstamo | Porcentaje (%) |
| loan_status | Numérica discreta | Estado de cumplimiento del préstamo (Variable objetivo) | 0 (Pagado / No default) / 1 (Incumplimiento / Default) |
| loan_percent_income | Numérica continua | Porcentaje del ingreso anual destinado a pagar el préstamo | Proporción (ej. 0.10 = 10%) |
| cb_person_default_on_file | Categórica | Indicador de si la persona tiene historial previo de impago | Y (Sí) / N (No) |
| cb_person_cred_hist_length | Numérica discreta | Años de historia crediticia de la persona | Años |

## Organización del repositorio
- analisis.py: solución final del proyecto.
- cuadernos/: cuadernos de exploración de Nicolas y Samuel.
- datos/: base de datos descargada de Kaggle.

## Conclusiones finales

**1. El grado del préstamo (Loan Grade) es el principal indicador de riesgo:**
Se evidenció una relación drástica entre la calificación del préstamo y la probabilidad de impago. Mientras que los préstamos de categorías A y B presentan tasas de default bajas (menores al 17%), el riesgo se dispara en las peores calificaciones: los préstamos de grado F superan el 70% de incumplimiento y los de grado G alcanzan un default casi absoluto (superior al 98%).

**2. La vulnerabilidad financiera de los arrendatarios:**
La situación de vivienda segmenta claramente el riesgo. Los clientes que viven en arriendo (RENT) tienen una tasa de impago que supera el 31%, lo que representa más del cuádruple de riesgo frente a quienes tienen vivienda propia totalmente pagada (OWN, con un ~7.5%).

**3. La liquidez actual importa más que la edad o el historial crediticio:**
Las variables con mayor correlación con el impago son el porcentaje del ingreso comprometido y la tasa de interés. Por el contrario, la edad de la persona y la proporción de su vida con historial crediticio mostraron una correlación casi nula. Esto demuestra que la carga financiera mensual del solicitante pesa mucho más en el riesgo que su edad o los años que lleve en el sistema financiero.

**4. El impacto del historial previo sobre el nivel de endeudamiento:**
El análisis agrupado demostró que los clientes que ya cuentan con un historial de impago en sus antecedentes (cb_person_default_on_file = Y) tienden a adquirir préstamos que comprometen una fracción mayor de sus ingresos anuales en comparación con quienes tienen su historial limpio. Esto refleja un comportamiento financiero donde el riesgo es compuesto: malos antecedentes sumados a una alta exposición salarial actual.