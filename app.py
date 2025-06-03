import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar dados
df = pd.read_excel("dados_rios_ficticios.xlsx")

# Classificação de risco com base no nível do rio
def classificar_nivel(nivel):
    if nivel < 5.5:
        return "Normal"
    elif 5.5 <= nivel < 7.5:
        return "Atenção"
    else:
        return "Risco"

df["nivel_categoria"] = df["nivel_metro"].apply(classificar_nivel)

# Engenharia de features
df["dia"] = df["data"].dt.day
df["mes"] = df["data"].dt.month
le = LabelEncoder()
df["rio_codificado"] = le.fit_transform(df["rio"])

# Features e alvo
X = df[["chuva_mm", "vazao_m3s", "dia", "mes", "rio_codificado"]]
y = df["nivel_categoria"]

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Dicionário de rios para mapeamento
mapa_rios = dict(zip(le.classes_, le.transform(le.classes_)))

# === Interface para input manual ===
print("🔍 Previsão de Risco do Nível do Rio")
print("Rios disponíveis:", ", ".join(mapa_rios.keys()))

rio = input("Digite o nome do rio: ")
while rio not in mapa_rios:
    rio = input("Nome inválido. Digite novamente: ")

chuva = float(input("Digite a quantidade de chuva (mm): "))
vazao = float(input("Digite a vazão do rio (m³/s): "))
dia = int(input("Digite o dia do mês (1-31): "))
mes = int(input("Digite o mês (1-12): "))

# Preparar dados para predição
entrada = pd.DataFrame([[
    chuva,
    vazao,
    dia,
    mes,
    mapa_rios[rio]
]], columns=["chuva_mm", "vazao_m3s", "dia", "mes", "rio_codificado"])

# Previsão
risco = modelo.predict(entrada)[0]
print(f"\n📢 Resultado: O nível do rio está em estado de **{risco.upper()}**.")
