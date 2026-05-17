# ============================================================
# CELL TAMBAHAN — SIMPAN MODEL UNTUK DEPLOYMENT
# Tambahkan cell ini di AKHIR notebook, jalankan setelah
# semua model selesai di-training
# ============================================================
import pickle, os
from google.colab import files

os.makedirs('models', exist_ok=True)

# Simpan ketiga model
with open('models/model_rf.pkl',  'wb') as f: pickle.dump(rf,      f)
with open('models/model_svm.pkl', 'wb') as f: pickle.dump(svm,     f)
with open('models/model_knn.pkl', 'wb') as f: pickle.dump(knn,     f)

# Simpan scaler dan label encoder (WAJIB untuk prediksi gambar baru)
with open('models/scaler.pkl',         'wb') as f: pickle.dump(scaler,  f)
with open('models/label_encoder.pkl',  'wb') as f: pickle.dump(le,      f)
with open('models/best_k.pkl',         'wb') as f: pickle.dump(best_k,  f)

print('[✓] Model berhasil disimpan!')
print(f'    model_rf.pkl       : {os.path.getsize("models/model_rf.pkl")/1024:.0f} KB')
print(f'    model_svm.pkl      : {os.path.getsize("models/model_svm.pkl")/1024:.0f} KB')
print(f'    model_knn.pkl      : {os.path.getsize("models/model_knn.pkl")/1024:.0f} KB')
print(f'    scaler.pkl         : {os.path.getsize("models/scaler.pkl")/1024:.0f} KB')
print(f'    label_encoder.pkl  : {os.path.getsize("models/label_encoder.pkl")/1024:.0f} KB')

# Download semua file ke komputer lokal
for fname in ['model_rf.pkl','model_svm.pkl','model_knn.pkl',
              'scaler.pkl','label_encoder.pkl','best_k.pkl']:
    files.download(f'models/{fname}')

print('\n[✓] Download selesai!')
print('Langkah selanjutnya:')
print('  1. Letakkan semua file .pkl ke dalam folder models/ di project Flask')
print('  2. Jalankan: python app.py')
print('  3. Buka browser: http://localhost:5000')
