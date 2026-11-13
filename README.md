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