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

# %% [markdown]
# ## 4. Valores inconsistentes y duplicados  (Integrante B)

# %% [markdown]
# ## 5. Creación de variables  (Integrante A)

# %% [markdown]
# ## 6. Filtrado y condiciones en pandas  (Integrante B)

# %% [markdown]
# ## 7. Agrupación con groupby  (Integrante A)

# %% [markdown]
# ## 8. Visualizaciones  (Integrante B)

# %% [markdown]
# ## 9. Conclusiones finales  (Integrante A)