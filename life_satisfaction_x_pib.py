import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import urllib.request
import os
# Para usar o algoritmo k-neighbors substitua o modelo importado
# import sklearn.neighbors

# Tweak para armazenar e ler os arquivos nesse diretorio
datapath = os.path.join("datasets", "lifesat", "")
print(datapath)


# Funcao para organizar os dados
def prepare_country_stats(oecd_bli, gdp_per_capita):
    """Combinado os dados de satisfacao de vida OCDE com os dados do PIB per capita"""
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(
        left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True
    )
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[
        keep_indices
    ]


# Baixa os dados
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
os.makedirs(datapath, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)

oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv(
    datapath + "gdp_per_capita.csv",
    thousands=",",
    delimiter="\t",
    encoding="latin1",
    na_values="n/a",
)

# Prepare os dados
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
y = np.c_[country_stats["Life satisfaction"]]
X = np.c_[country_stats["GDP per capita"]]

# Vizualize os dados
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
plt.show()

# Selecione o modelo de regressao linear
model = sklearn.linear_model.LinearRegression()
# use model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# para trocar para k-neighbors algorithm
model.fit(X, y)

# Efetua uma predicao para o chipre
X_new = [[22587]]  # Pib per capita do chipre
print(model.predict(X_new))
