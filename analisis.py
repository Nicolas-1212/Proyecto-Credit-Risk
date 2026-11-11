# %% [markdown]
# # Título del proyecto
# Integrantes: Nombre A, Nombre B

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

# %% [markdown]
# ## 3. Identificación de outliers  (Integrante A)
columnas_numericas = df.select_dtypes(include=[np.number]).columns
num_variables = len(columnas_numericas)
num_filas = (num_variables + 2) // 3
fig, axes = plt.subplots(num_filas, 3, figsize=(15, 5 * num_filas))
axes = axes.flatten()
for i, col in enumerate(columnas_numericas):
    sns.boxplot(y=df[col], ax=axes[i], color='royalblue')
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_ylabel('')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
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