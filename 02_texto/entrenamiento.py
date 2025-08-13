import nltk
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)

nltk.download("stopwords", download_dir=nltk_data_path)

try:
    df = pd.read_csv("noticias.csv")
    if "titulo" in df.columns:
        titulos_noticias = df["titulo"].astype(str).tolist()
    else:
        titulos_noticias = df[df.columns[0]].astype(str).tolist()
except pd.errors.ParserError:
    df = pd.read_csv("noticias.csv", header=None)
    titulos_noticias = df[0].astype(str).tolist()

spanish_stopwords = stopwords.words("spanish")

vectorizador = TfidfVectorizer(stop_words=spanish_stopwords)
X = vectorizador.fit_transform(titulos_noticias)

modelo = KMeans(n_clusters=4, random_state=1234, n_init=10)
modelo.fit(X)

joblib.dump(modelo, "modelo_texto.pkl")

print(f"Clusters asignados: {modelo.labels_}")

for i, texto in enumerate(titulos_noticias):
    print(f"{texto} ---> Cluster {modelo.labels_[i]}")