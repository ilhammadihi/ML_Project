import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🤖 LOGISTIC REGRESSION")
print("=" * 60)

# 1. Charge les données
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"\n✅ Données chargées")
print(f"   Train : {X_train.shape}")
print(f"   Test  : {X_test.shape}")

# 2. Entraînement
print("\n⏳ Entraînement...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✅ Entraînement terminé")

# 3. Prédictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 4. Évaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 RÉSULTATS :")
print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Rapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['Sain', 'Malade']))

# 5. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sain', 'Malade'],
            yticklabels=['Sain', 'Malade'])
plt.title('Matrice de Confusion - Logistic Regression')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.savefig('images/confusion_matrix_lr.png')
plt.show()

print("\n✅ Graphique sauvegardé : images/confusion_matrix_lr.png")

# 6. Sauvegarde le modèle
joblib.dump(model, 'models/logistic_regression.pkl')
print("💾 Modèle sauvegardé : models/logistic_regression.pkl")

print("\n✅ Logistic Regression terminé !")