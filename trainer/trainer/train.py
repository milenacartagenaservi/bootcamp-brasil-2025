import pandas as pd
import xgboost as xgb
import joblib
import os
from google.cloud import storage

# 🔽 Função para baixar o CSV do GCS para o disco local (/tmp)
def download_csv_from_gcs(bucket_name, source_blob, dest_file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_file)

# 🛠️ Função de engenharia de atributos
def create_features(df):
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week")
    df["week_number"] = (df["week"] - df["week"].min()).dt.days
    df["week_of_year"] = df["week"].dt.week  # compatível com Python 3.7
    df["lag_1"] = df["weekly_sales"].shift(1)
    df["lag_2"] = df["weekly_sales"].shift(2)
    df["lag_3"] = df["weekly_sales"].shift(3)
    df = df.dropna().reset_index(drop=True)
    return df

# 🚀 Função principal
def main():
    bucket_name = "bootcamp-brasil-central1-v1"
    blob_path = "data/weekly_sales.csv"
    local_path = "/tmp/weekly_sales.csv"

    download_csv_from_gcs(bucket_name, blob_path, local_path)
    df = pd.read_csv(local_path)
    df = create_features(df)

    X = df[["lag_1", "lag_2", "lag_3", "week_of_year"]]
    y = df["weekly_sales"]

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    local_model_path = "/tmp/model.bst"
    joblib.dump(model, local_model_path)

    output_dir = os.environ["AIP_MODEL_DIR"].rstrip("/")
    model_filename = "model.bst"

    # Subir archivo local al bucket de GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{output_dir.split('gs://'+bucket_name+'/')[1]}/{model_filename}")
    blob.upload_from_filename(local_model_path)

    print(f"✅ Modelo exportado en GCS: {output_dir}/{model_filename}")


# 🔁 Executa se for o script principal
if __name__ == "__main__":
    main()
