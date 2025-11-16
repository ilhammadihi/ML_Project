import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🌲 RANDOM FOREST")
print("=" * 60)

# 1. Charge les données
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"\n✅ Données chargées")

# 2. Entraînement
print("\n⏳ Entraînement...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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

# 5. Feature Importance
print("\n🔥 TOP 5 Features importantes :")
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in importances.head(5).iterrows():
    print(f"   {i+1}. {row['Feature']:15s} : {row['Importance']:.4f}")

# 6. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Sain', 'Malade'],
            yticklabels=['Sain', 'Malade'])
plt.title('Matrice de Confusion - Random Forest')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.savefig('images/confusion_matrix_rf.png')
plt.show()

print("\n✅ Graphique sauvegardé : images/confusion_matrix_rf.png")

# 7. Sauvegarde
joblib.dump(model, 'models/random_forest.pkl')
print("💾 Modèle sauvegardé : models/random_forest.pkl")

print("\n✅ Random Forest terminé !")