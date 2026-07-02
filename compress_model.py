import joblib
import os

model = joblib.load("readmitted_model.pkl")

joblib.dump(
    model,
    "readmitted_model_compressed.pkl",
    compress=("lzma", 9)
)

print(os.path.getsize("readmitted_model_compressed.pkl")/1024/1024)