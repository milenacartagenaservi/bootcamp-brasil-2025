from setuptools import setup  # 📦 Utilitário para empacotar e instalar o módulo como pacote Python

# 📌 Definição do pacote "trainer"

setup(
    name="trainer",           # Nome do pacote
    version="0.1",            # Versão do pacote
    packages=["trainer"],     # Pasta que contém o código-fonte (trainer/train.py)

    # 📦 Dependências necessárias para rodar o script
    install_requires=[
        "pandas",               # Manipulação de dados
        "xgboost",              # Algoritmo de regressão
        "joblib",               # Para salvar e carregar modelos
        "google-cloud-storage"  # Para baixar arquivos do GCS
    ],

    # 🚀 Ponto de entrada para execução no Vertex AI
    entry_points={
        "console_scripts": ["train=trainer.train:main"],  # Executa trainer/train.py:main como script
    },
)
