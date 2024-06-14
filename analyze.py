import joblib
from sklearn.preprocessing import LabelEncoder

# Muat model dari file
model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)

def predict(data):
    # Lakukan prediksi dengan model yang telah dimuat
    pred = model.predict(data)
    # Dekode label prediksi menjadi bentuk aslinya
    predicted_label = pred[0]
    if predicted_label == 0:
        predicted_tone = 'Summer'
    elif predicted_label == 1:
        predicted_tone = 'Winter'  
    elif predicted_label == 2:
        predicted_tone = 'Autumn'
    elif predicted_label == 3:
        predicted_tone = 'Spring'
    return predicted_tone