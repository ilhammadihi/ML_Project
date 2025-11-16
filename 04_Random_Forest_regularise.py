"""
Heart Disease Prediction - Random Forest RÉGULARISÉ
===================================================
Version avec contraintes pour éviter l'overfitting
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("🌲 RANDOM FOREST - VERSION RÉGULARISÉE")
print("=" * 60)

# ====================================
# 1. CHARGEMENT DES DONNÉES
# ====================================

print("\n📥 Chargement des données...")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

print(f"✅ Train : {X_train.shape}")
print(f"✅ Test  : {X_test.shape}")

# ====================================
# 2. ENTRAÎNEMENT AVEC RÉGULARISATION
# ====================================

print("\n⏳ Entraînement avec contraintes de régularisation...")

# Paramètres CONTRAINTS pour éviter overfitting
model_regularized = RandomForestClassifier(
    n_estimators=50,              # Moins d'arbres (vs 100)
    max_depth=5,                  # Limite la profondeur ⚠️ CLÉ
    min_samples_split=10,         # Min échantillons pour splitter (vs 2)
    min_samples_leaf=5,           # Min échantillons par feuille (vs 1)
    max_features='sqrt',          # Moins de features par arbre
    random_state=42,
    n_jobs=-1
)

print("\n📊 Paramètres RÉGULARISÉS :")
print(f"   - n_estimators      : {model_regularized.n_estimators} (réduit)")
print(f"   - max_depth         : {model_regularized.max_depth} (limité)")
print(f"   - min_samples_split : {model_regularized.min_samples_split} (augmenté)")
print(f"   - min_samples_leaf  : {model_regularized.min_samples_leaf} (augmenté)")
print(f"   - max_features      : {model_regularized.max_features}")

model_regularized.fit(X_train, y_train)
print("✅ Entraînement terminé")

# ====================================
# 3. COMPARAISON : AVANT vs APRÈS
# ====================================

print("\n" + "=" * 60)
print("📊 COMPARAISON : MODÈLE ORIGINAL VS RÉGULARISÉ")
print("=" * 60)

# Charge le modèle original (si existe)
try:
    model_original = joblib.load('models/random_forest.pkl')
    
    # Prédictions original
    y_train_pred_orig = model_original.predict(X_train)
    y_test_pred_orig = model_original.predict(X_test)
    
    train_acc_orig = accuracy_score(y_train, y_train_pred_orig)
    test_acc_orig = accuracy_score(y_test, y_test_pred_orig)
    
    print("\n🌲 MODÈLE ORIGINAL (Sans contraintes) :")
    print(f"   Train Accuracy : {train_acc_orig:.4f} ({train_acc_orig*100:.2f}%)")
    print(f"   Test Accuracy  : {test_acc_orig:.4f} ({test_acc_orig*100:.2f}%)")
    print(f"   Différence     : {(train_acc_orig - test_acc_orig):.4f}")
    
    if test_acc_orig >= 0.95:
        print("   ⚠️  Performance suspicieuse (>=95%)")

except:
    print("\n⚠️  Modèle original non trouvé")
    model_original = None

# Prédictions régularisé
y_train_pred_reg = model_regularized.predict(X_train)
y_test_pred_reg = model_regularized.predict(X_test)
y_test_proba_reg = model_regularized.predict_proba(X_test)[:, 1]

train_acc_reg = accuracy_score(y_train, y_train_pred_reg)
test_acc_reg = accuracy_score(y_test, y_test_pred_reg)

print("\n🌲 MODÈLE RÉGULARISÉ (Avec contraintes) :")
print(f"   Train Accuracy : {train_acc_reg:.4f} ({train_acc_reg*100:.2f}%)")
print(f"   Test Accuracy  : {test_acc_reg:.4f} ({test_acc_reg*100:.2f}%)")
print(f"   Différence     : {(train_acc_reg - test_acc_reg):.4f}")

if train_acc_reg - test_acc_reg < 0.1:
    print("   ✅ Pas d'overfitting significatif")
else:
    print("   ⚠️  Overfitting détecté")

# ====================================
# 4. VALIDATION CROISÉE
# ====================================

print("\n" + "=" * 60)
print("🔄 VALIDATION CROISÉE (Plus Fiable)")
print("=" * 60)

# Combine train + test
X_full = pd.concat([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

print("\n⏳ Cross-validation 5-fold...")
cv_scores = cross_val_score(model_regularized, X_full, y_full, 
                            cv=5, scoring='accuracy')

print(f"\n📊 Scores par fold :")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i} : {score:.4f} ({score*100:.2f}%)")

mean_cv = cv_scores.mean()
std_cv = cv_scores.std()

print(f"\n📈 Statistiques CV :")
print(f"   Moyenne    : {mean_cv:.4f} ({mean_cv*100:.2f}%)")
print(f"   Écart-type : {std_cv:.4f}")
print(f"   Min        : {cv_scores.min():.4f}")
print(f"   Max        : {cv_scores.max():.4f}")

print("\n💡 Le score CV est plus représentatif que le test accuracy !")

# ====================================
# 5. MÉTRIQUES DÉTAILLÉES
# ====================================

print("\n" + "=" * 60)
print("📊 MÉTRIQUES DÉTAILLÉES")
print("=" * 60)

accuracy = accuracy_score(y_test, y_test_pred_reg)
precision = precision_score(y_test, y_test_pred_reg)
recall = recall_score(y_test, y_test_pred_reg)
f1 = f1_score(y_test, y_test_pred_reg)
roc_auc = roc_auc_score(y_test, y_test_proba_reg)

print(f"\n📈 Performances sur Test Set :")
print(f"   Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
print(f"   ROC-AUC   : {roc_auc:.4f}")

print("\n📋 Rapport de Classification :")
print(classification_report(y_test, y_test_pred_reg, 
                           target_names=['Sain', 'Malade']))

# ====================================
# 6. MATRICE DE CONFUSION
# ====================================

cm = confusion_matrix(y_test, y_test_pred_reg)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Sain', 'Malade'],
            yticklabels=['Sain', 'Malade'],
            cbar_kws={'label': 'Nombre'})

plt.title('Matrice de Confusion - Random Forest Régularisé', 
          fontsize=14, fontweight='bold')
plt.ylabel('Vraie Classe', fontsize=12)
plt.xlabel('Classe Prédite', fontsize=12)

tn, fp, fn, tp = cm.ravel()
plt.text(0.5, -0.15, f'TN={tn} | FP={fp} | FN={fn} | TP={tp}', 
         ha='center', transform=plt.gca().transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('images/confusion_matrix_rf_reg.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Matrice sauvegardée : images/confusion_matrix_rf_reg.png")

print(f"\n🔍 Erreurs commises : {fp + fn} / {len(y_test)}")
print(f"   False Positives : {fp}")
print(f"   False Negatives : {fn}")

# ====================================
# 7. FEATURE IMPORTANCE
# ====================================

print("\n🔥 Feature Importance :")
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model_regularized.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in importances.head(5).iterrows():
    print(f"   {i+1}. {row['Feature']:15s} : {row['Importance']:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'], importances['Importance'], 
         color=plt.cm.viridis(np.linspace(0, 1, len(importances))))
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest Régularisé', fontweight='bold')
plt.tight_layout()
plt.savefig('images/feature_importance_rf_reg.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature importance sauvegardée : images/feature_importance_rf_reg.png")

# ====================================
# 8. SAUVEGARDE
# ====================================

print("\n💾 Sauvegarde du modèle régularisé...")
joblib.dump(model_regularized, 'models/random_forest_regularized.pkl')
print("✅ Modèle sauvegardé : models/random_forest_regularized.pkl")

# Sauvegarde les métriques
metrics = {
    'Model': 'Random Forest Regularized',
    'Test_Accuracy': accuracy,
    'CV_Mean': mean_cv,
    'CV_Std': std_cv,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc,
    'Train_Accuracy': train_acc_reg,
    'Overfitting': train_acc_reg - test_acc_reg
}

pd.DataFrame([metrics]).to_csv('models/rf_regularized_metrics.csv', index=False)
print("✅ Métriques sauvegardées : models/rf_regularized_metrics.csv")

# ====================================
# RÉSUMÉ & RECOMMANDATIONS
# ====================================

print("\n" + "=" * 60)
print("💡 RÉSUMÉ & RECOMMANDATIONS")
print("=" * 60)

print(f"\n📊 SCORES À UTILISER DANS TON PROJET :")
print(f"   Test Accuracy       : {accuracy*100:.2f}%")
print(f"   CV Accuracy (5-fold): {mean_cv*100:.2f}% ± {std_cv*100:.2f}%")
print(f"   ROC-AUC             : {roc_auc:.4f}")

print("\n🎯 QUEL SCORE PRÉSENTER ?")
if mean_cv < 0.95:
    print("   ✅ Utilise le score CV (plus crédible)")
    print(f"   → \"Random Forest avec CV : {mean_cv*100:.2f}%\"")
else:
    print("   ⚠️  Même le CV est très élevé")
    print("   → Mentionne que c'est un dataset simple/petit")

print("\n💬 DANS TA PRÉSENTATION, DIS :")
print("   1. 'Dataset petit (303 patients) → performances élevées'")
print("   2. 'Validation croisée utilisée pour éviter surestimation'")
print("   3. 'Régularisation appliquée (max_depth=5)'")
print("   4. 'Future: Valider sur données externes'")

print("\n✅ Random Forest Régularisé terminé !")