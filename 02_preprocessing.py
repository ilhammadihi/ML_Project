import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 60)
print("🔧 PREPROCESSING")
print("=" * 60)

# 1. Charge les données
df = pd.read_csv('data/heart.csv')
print(f"\n✅ Dataset chargé : {df.shape}")

# 2. Sépare X et y
X = df.drop('target', axis=1)
y = df['target']

print(f"\n✅ Features (X) : {X.shape}")
print(f"✅ Target (y) : {y.shape}")

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"\n📊 Train : {X_train.shape[0]} échantillons (80%)")
print(f"📊 Test  : {X_test.shape[0]} échantillons (20%)")

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reconvertit en DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\n✅ Feature scaling appliqué")

# 5. Sauvegarde
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

X_train_scaled.to_csv('data/X_train.csv', index=False)
X_test_scaled.to_csv('data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)
joblib.dump(scaler, 'models/scaler.pkl')

print("\n💾 Fichiers sauvegardés :")
print("   - data/X_train.csv")
print("   - data/X_test.csv")
print("   - data/y_train.csv")
print("   - data/y_test.csv")
print("   - models/scaler.pkl")

print("\n✅ Preprocessing terminé !")