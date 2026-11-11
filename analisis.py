# %% [markdown]
# # Proyecto-Credit-Risk
# Integrantes: Nicolas Ramirez (A), Samuel Arteaga (B)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datos/credit_risk_dataset.csv")

# %% [markdown]
# ## 1. Carga e inspección inicial de los datos  (Integrante A)
print(f"Dimensiones del dataset: filas: {df.shape[0]}, columnas: {df.shape[1]}")
print(df.dtypes)
print(df.describe(include='all'))
# %% [markdown]
# ## 2. Manejo de datos faltantes  (Integrante B)
##Datos faltantes
df_limpio = df.copy()

faltantes = pd.DataFrame({
    "faltantes": df_limpio.isna().sum(),
    "porcentaje": (df_limpio.isna().mean() * 100).round(2),
})
faltantes = faltantes[faltantes["faltantes"] > 0]
print(faltantes)

cols_nulos = faltantes.index.tolist()
print("\nFilas con al menos un faltante:", df_limpio.isna().any(axis=1).sum())
print("Filas con ambos faltantes:", df_limpio[cols_nulos].isna().all(axis=1).sum())

#Datos imposibles en el contexto
edades_raras = (df["person_age"] > 100).sum()
antig_rara = (df["person_emp_length"] > 60).sum()
print(f"Personas con edad mayor a 100 años: {edades_raras}")
print(f"Personas con más de 60 años de antigüedad laboral: {antig_rara}")
 
df2 = df.copy()
df2.loc[df2["person_age"] > 100, "person_age"] = np.nan
df2.loc[df2["person_emp_length"] > 60, "person_emp_length"] = np.nan
print("\nEstos valores imposibles se convirtieron en datos faltantes (NaN).")

#Imputar datos faltantes o imposibles
df_limpio["loan_int_rate"] = df_limpio["loan_int_rate"].fillna(
    df_limpio.groupby("loan_grade")["loan_int_rate"].transform("median"))

df_limpio["person_emp_length"] = df_limpio["person_emp_length"].fillna(
    df_limpio.groupby("person_home_ownership")["person_emp_length"].transform("median"))

df_limpio["person_age"] = df_limpio["person_age"].fillna(df_limpio["person_age"].median())

##Resultado despues de imputar
print("Datos faltantes por columna después de imputar:")
print(df_limpio.isna().sum())
print("\nTotal de datos faltantes:", df_limpio.isna().sum().sum())

print("\nPrimeras 5 filas del dataframe limpio:")
print(df_limpio.head())
print("\nDimensiones del dataframe limpio:", df_limpio.shape)

##Mapa de Calor
corr_matrix = df.select_dtypes(include=['number']).corr()
nulos_df = pd.DataFrame(
    {
        'Faltantes': df.isnull().sum(),
        'Porcentaje (%)': (df.isnull().mean() * 100).round(2),
    }
)
print(nulos_df[nulos_df['Faltantes'] > 0])

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5
)
plt.title('Matriz de Correlación de Variables Numéricas', fontsize=14, pad=15)
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 3. Identificación de outliers  (Integrante A)
columnas_numericas = df_limpio.select_dtypes(include=[np.number]).columns
num_variables = len(columnas_numericas)
num_filas = (num_variables + 2) // 3
fig, axes = plt.subplots(num_filas, 3, figsize=(15, 5 * num_filas))
axes = axes.flatten()
for i, col in enumerate(columnas_numericas):
    sns.boxplot(y=df_limpio[col], ax=axes[i], color='royalblue')
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_ylabel('')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 4. Valores inconsistentes y duplicados  (Integrante B)
# Valores inconsistentes en texto
cols_texto = df_limpio.select_dtypes(include=["object", "string"]).columns.tolist()
print("Columnas de texto:", cols_texto)

#Estandarizacion de el dataframe
antes = df_limpio[cols_texto].copy()

for col in cols_texto:
    df_limpio[col] = df_limpio[col].str.strip().str.upper()

cambios = (antes != df_limpio[cols_texto]).sum()
print("Valores modificados por columna al estandarizar:")
print(cambios)

##Analisis de los  duplicados
filas_antes = len(df_limpio)
n_duplicados = df_limpio.duplicated().sum()
print(f"Filas duplicadas exactas: {n_duplicados} "
      f"({n_duplicados / filas_antes * 100:.2f}%)")

## Eliminar duplicados
df_limpio = df_limpio.drop_duplicates(keep="first")

print(f"\nFilas antes:   {filas_antes}")
print(f"Filas después: {len(df_limpio)}")
print(f"Filas eliminadas: {filas_antes - len(df_limpio)}")
print("Duplicados restantes:", df_limpio.duplicated().sum())

print("\nLa decisión es eliminar los duplicados exactos: son registros repetidos "
      "del mismo préstamo y dejarlos sesgaría el análisis y los modelos.")
print("Dimensiones de df_limpio:", df_limpio.shape)


# %% [markdown]
# ## 5. Creación de variables  (Integrante A)
# Variable 1: Proporción del historial crediticio respecto a la edad
# ¿Qué porcentaje de la vida de la persona abarca su historial de crédito?
df_limpio['historial_vs_edad'] = df['cb_person_cred_hist_length'] / df['person_age']
# Variable 2: Monto de interés anual estimado
# Calcula cuánto dinero representa el interés del préstamo 
# (Monto del préstamo * (Tasa de interés / 100))
df_limpio['interes_estimado'] = df['loan_amnt'] * (df['loan_int_rate'] / 100)
# %% [markdown]
# ## 6. Filtrado y condiciones en pandas  (Integrante B)
# Filtrado y condiciones en pandas
tasa_global = df_limpio["loan_status"].mean() * 100
print(f"Tasa de incumplimiento global: {tasa_global:.2f}%")

#  1. Préstamos con interés estimado muy alto (percentil 90) 
p90 = df_limpio["interes_estimado"].quantile(0.90)
f1 = df_limpio[df_limpio["interes_estimado"] > p90]
tasa1 = f1["loan_status"].mean() * 100
print(f"\n1) Interés estimado > {p90:,.0f}: {len(f1)} préstamos, incumplimiento {tasa1:.2f}%")
print(f"   Conclusión: los préstamos con mayor costo en intereses incumplen "
      f"{'más' if tasa1 > tasa_global else 'menos'} que el promedio ({tasa_global:.2f}%).")

#  2. Historial crediticio largo respecto a la edad 
f2 = df_limpio[df_limpio["historial_vs_edad"] > 0.30]
tasa2 = f2["loan_status"].mean() * 100
print(f"\n2) Historial > 30% de la edad: {len(f2)} personas, incumplimiento {tasa2:.2f}%")
print(f"   Conclusión: tener un historial largo para su edad "
      f"{'no reduce' if tasa2 >= tasa_global else 'reduce'} el incumplimiento frente al promedio.")

# 3. Registros inconsistentes (edad o historial imposibles) 
f3 = df_limpio[(df_limpio["person_age"] > 100) |
               (df_limpio["historial_vs_edad"] >= 1) |
               (df_limpio["person_emp_length"] > df_limpio["person_age"] - 14)]
print(f"\n3) Registros con edad/antigüedad/historial imposibles: {len(f3)}")
print(f3[["person_age", "person_emp_length", "cb_person_cred_hist_length"]].head())
print("   Conclusión: son errores de captura que conviene revisar o excluir antes de modelar.")

# 4. Arrendatarios con carga alta sobre el ingreso 
f4 = df_limpio[(df_limpio["person_home_ownership"] == "RENT") &
               (df_limpio["loan_percent_income"] > 0.40)]
tasa4 = f4["loan_status"].mean() * 100
print(f"\n4) RENT y préstamo > 40% del ingreso: {len(f4)} personas, incumplimiento {tasa4:.2f}%")
print(f"   Conclusión: quienes arriendan y deben mucho respecto a su ingreso "
      f"tienen incumplimiento de {tasa4:.2f}%, {'superior' if tasa4 > tasa_global else 'inferior'} al global.")

# 5. Historial corto y monto de préstamo alto 
mediana_monto = df_limpio["loan_amnt"].median()
f5 = df_limpio[(df_limpio["historial_vs_edad"] < 0.10) &
               (df_limpio["loan_amnt"] > mediana_monto)]
tasa5 = f5["loan_status"].mean() * 100
dif5 = tasa5 - tasa_global
print(f"\n5) Historial < 10% de la edad y monto > {mediana_monto:,.0f}: {len(f5)} préstamos, incumplimiento {tasa5:.2f}%")
print(f"   Conclusión: la tasa de incumplimiento de este grupo es {tasa5:.2f}% frente a "
      f"{tasa_global:.2f}% del total, es decir, {abs(dif5):.2f} puntos "
      f"{'por encima' if dif5 > 0 else 'por debajo'} de la tasa global. "
      f"{'Pedir montos altos con poca trayectoria crediticia es más riesgoso que el promedio.' if dif5 > 0 else 'Pedir montos altos con poca trayectoria crediticia es menos riesgoso que el promedio.'}")

# %% [markdown]
# ## 7. Agrupación con groupby  (Integrante A)

# %% [markdown]
# ## 8. Visualizaciones  (Integrante B)
sns.set_theme(style="whitegrid")
# 1. DISPERSIÓN: ingreso vs monto del préstamo 
muestra = df_limpio[df_limpio["person_income"] < df_limpio["person_income"].quantile(0.99)]
plt.figure(figsize=(8, 5))
sns.scatterplot(data=muestra, x="person_income", y="loan_amnt", alpha=0.3, color="steelblue")
plt.title("Ingreso vs monto del préstamo")
plt.xlabel("Ingreso anual")
plt.ylabel("Monto del préstamo")
plt.tight_layout()
plt.show()

corr1 = df_limpio["person_income"].corr(df_limpio["loan_amnt"])
print(f"Conclusión: la correlación entre ingreso y monto es {corr1:.2f}, "
      f"{'una relación positiva moderada: quien gana más tiende a pedir más.' if corr1 > 0.3 else 'una relación débil: el monto pedido casi no depende del ingreso.'}")

# 2. BARRAS AGRUPADAS: monto promedio por motivo y tipo de vivienda
tabla2 = df_limpio.groupby(["loan_intent", "person_home_ownership"])["loan_amnt"].mean().unstack()
tabla2.plot(kind="bar", figsize=(10, 5))
plt.title("Monto promedio del préstamo por motivo y tipo de vivienda")
plt.xlabel("Motivo del préstamo")
plt.ylabel("Monto promedio")
plt.xticks(rotation=30)
plt.legend(title="Vivienda")
plt.tight_layout()
plt.show()

prom2 = df_limpio.groupby("loan_intent")["loan_amnt"].mean()
print(f"Conclusión: el motivo con mayor monto promedio es {prom2.idxmax()} ({prom2.max():,.0f}) "
      f"y el de menor monto es {prom2.idxmin()} ({prom2.min():,.0f}).")

# 3. CIRCULAR: distribución por tipo de vivienda
conteo3 = df_limpio["person_home_ownership"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(conteo3, labels=conteo3.index, autopct="%1.1f%%", startangle=90)
plt.title("Distribución por tipo de vivienda")
plt.tight_layout()
plt.show()

print(f"Conclusión: {conteo3.index[0]} es el tipo de vivienda más común "
      f"({conteo3.iloc[0] / conteo3.sum() * 100:.1f}% de las personas) y "
      f"{conteo3.index[-1]} el menos común ({conteo3.iloc[-1] / conteo3.sum() * 100:.1f}%).")
# 4. Histograma: distribución del monto del préstamo
media4 = df_limpio["loan_amnt"].mean()
mediana4 = df_limpio["loan_amnt"].median()

plt.figure(figsize=(8, 5))
sns.histplot(data=df_limpio, x="loan_amnt", bins=20, color="steelblue")
plt.axvline(media4, color="red", linestyle="--", label=f"Media: {media4:,.0f}")
plt.axvline(mediana4, color="black", linestyle=":", label=f"Mediana: {mediana4:,.0f}")
plt.title("Distribución del monto del préstamo")
plt.xlabel("Monto del préstamo")
plt.ylabel("Cantidad de préstamos")
plt.legend()
plt.tight_layout()
plt.show()

# 5. BARRAS: incumplimiento por rango de edad 
df_limpio["rango_edad"] = pd.cut(df_limpio["person_age"], bins=[0, 25, 30, 40, 50, 120],
                                 labels=["<=25", "26-30", "31-40", "41-50", ">50"])
tasa5 = df_limpio.groupby("rango_edad", observed=True)["loan_status"].mean().mul(100)
plt.figure(figsize=(8, 5))
tasa5.plot(kind="bar", color="slateblue")
plt.axhline(tasa_global, color="black", linestyle="--", label="Tasa global")
plt.title("Tasa de incumplimiento por rango de edad")
plt.xlabel("Rango de edad")
plt.ylabel("Incumplimiento (%)")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Conclusión: el rango {tasa5.idxmax()} tiene el mayor incumplimiento ({tasa5.max():.2f}%) "
      f"y el rango {tasa5.idxmin()} el menor ({tasa5.min():.2f}%), con {tasa_global:.2f}% a nivel global. "
      f"La diferencia entre edades es de {tasa5.max() - tasa5.min():.2f} puntos.")


# %% [markdown]
# ## 9. Conclusiones finales  (Integrante A)