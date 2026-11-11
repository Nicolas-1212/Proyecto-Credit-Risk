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

# %% [markdown]
# ## 7. Agrupación con groupby  (Integrante A)

# %% [markdown]
# ## 8. Visualizaciones  (Integrante B)

# %% [markdown]
# ## 9. Conclusiones finales  (Integrante A)