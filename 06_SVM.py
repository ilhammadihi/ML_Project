"""
Heart Disease Prediction - Support Vector Machine (SVM)
=======================================================
Modèle 3 : SVM avec kernel RBF
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ CRÉÉ LES DOSSIERS
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("🎯 SUPPORT VECTOR MACHINE (SVM)")
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

print("\n💡 Note : SVM nécessite des données normalisées")
print("   (déjà fait dans le preprocessing avec StandardScaler)")

# ====================================
# 2. ENTRAÎNEMENT DU MODÈLE
# ====================================

print("\n⏳ Entraînement du SVM...")

# SVM avec kernel RBF (non-linéaire)
model = SVC(
    kernel='rbf',           # Kernel Radial Basis Function (non-linéaire)
    C=1.0,                  # Paramètre de régularisation
    gamma='scale',          # Coefficient du kernel
    probability=True,       # Active les probabilités (pour ROC-AUC)
    random_state=42
)

print("\n📊 Paramètres du modèle :")
print(f"   - Kernel       : {model.kernel}")
print(f"   - C (régul.)   : {model.C}")
print(f"   - Gamma        : {model.gamma}")
print(f"   - Probability  : {model.probability}")

model.fit(X_train, y_train)
print("✅ Entraînement terminé")

# ====================================
# 3. PRÉDICTIONS
# ====================================

print("\n🎯 Prédictions...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Probabilités (nécessite probability=True)
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# ====================================
# 4. ÉVALUATION DES PERFORMANCES
# ====================================

print("\n" + "=" * 60)
print("📊 ÉVALUATION DES PERFORMANCES")
print("=" * 60)

# Métriques Train
train_acc = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print(f"\n📊 TRAIN SET :")
print(f"   Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   ROC-AUC  : {train_auc:.4f}")

# Métriques Test
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n📊 TEST SET :")
print(f"   Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision : {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall    : {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
print(f"   ROC-AUC   : {roc_auc:.4f}")

# Check overfitting
diff = train_acc - accuracy
print(f"\n🔍 Analyse Overfitting :")
print(f"   Différence Train-Test : {diff:.4f} ({diff*100:.2f}%)")
if diff > 0.1:
    print("   ⚠️  Overfitting détecté")
elif diff > 0.05:
    print("   ⚠️  Léger overfitting")
else:
    print("   ✅ Pas d'overfitting significatif")

# Rapport détaillé
print("\n📋 Rapport de Classification Détaillé :")
print(classification_report(y_test, y_test_pred, 
                           target_names=['Sain (0)', 'Malade (1)']))

# ====================================
# 5. VALIDATION CROISÉE
# ====================================

print("\n" + "=" * 60)
print("🔄 VALIDATION CROISÉE (5-FOLD)")
print("=" * 60)

# Combine train + test
X_full = pd.concat([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

print("\n⏳ Cross-validation en cours...")
cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')

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

print("\n💡 Le score CV est la métrique la plus fiable !")

# ====================================
# 6. MATRICE DE CONFUSION
# ====================================

print("\n📊 Génération de la matrice de confusion...")
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Sain', 'Malade'],
            yticklabels=['Sain', 'Malade'],
            cbar_kws={'label': 'Nombre de prédictions'})

plt.title('Matrice de Confusion - SVM', fontsize=14, fontweight='bold')
plt.ylabel('Vraie Classe', fontsize=12)
plt.xlabel('Classe Prédite', fontsize=12)

# Annotations
tn, fp, fn, tp = cm.ravel()
plt.text(0.5, -0.15, f'TN = {tn}  |  FP = {fp}', 
         ha='center', transform=plt.gca().transAxes, fontsize=10)
plt.text(0.5, -0.20, f'FN = {fn}  |  TP = {tp}', 
         ha='center', transform=plt.gca().transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('images/confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Matrice sauvegardée : images/confusion_matrix_svm.png")

print(f"\n🔍 Interprétation :")
print(f"   True Negatives  : {tn} (Sains correctement identifiés)")
print(f"   False Positives : {fp} (Sains prédits malades)")
print(f"   False Negatives : {fn} (Malades prédits sains) ⚠️")
print(f"   True Positives  : {tp} (Malades correctement identifiés)")
print(f"\n   Erreurs totales : {fp + fn} / {len(y_test)}")

# ====================================
# 7. COURBE ROC
# ====================================

print("\n📈 Génération de la courbe ROC...")
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'SVM (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
plt.title('Courbe ROC - SVM', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/roc_curve_svm.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Courbe ROC sauvegardée : images/roc_curve_svm.png")

# ====================================
# 8. VECTEURS DE SUPPORT
# ====================================

print("\n" + "=" * 60)
print("🎯 ANALYSE DES VECTEURS DE SUPPORT")
print("=" * 60)

n_support = model.n_support_
print(f"\n📊 Nombre de vecteurs de support :")
print(f"   Classe 0 (Sain)   : {n_support[0]}")
print(f"   Classe 1 (Malade) : {n_support[1]}")
print(f"   Total             : {sum(n_support)}")

pct_support = (sum(n_support) / len(X_train)) * 100
print(f"\n   Pourcentage du train set : {pct_support:.2f}%")

if pct_support < 30:
    print("   ✅ Bon - Peu de vecteurs de support")
elif pct_support < 50:
    print("   ⚠️  Modéré - Beaucoup de vecteurs")
else:
    print("   ⚠️  Élevé - Possible sur-complexité")

# ====================================
# 9. OPTIMISATION HYPERPARAMÈTRES (OPTIONNEL)
# ====================================

print("\n" + "=" * 60)
print("⚙️  OPTIMISATION DES HYPERPARAMÈTRES (Optionnel)")
print("=" * 60)

print("\n💡 Pour améliorer les performances, tu peux optimiser C et gamma")
print("   Attention : Cela prend du temps (~2-3 minutes)")

optimize = input("\n   Lancer l'optimisation ? (o/n) : ").lower() == 'o'

if optimize:
    print("\n⏳ GridSearchCV en cours...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n✅ Optimisation terminée !")
    print(f"\n🏆 Meilleurs paramètres :")
    for param, value in grid_search.best_params_.items():
        print(f"   {param} : {value}")
    
    print(f"\n📊 Meilleur score CV : {grid_search.best_score_:.4f}")
    
    # Utilise le meilleur modèle
    model = grid_search.best_estimator_
    
    # Réévalue
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy_opt = accuracy_score(y_test, y_test_pred)
    print(f"📈 Test accuracy avec params optimisés : {accuracy_opt:.4f}")
else:
    print("   ⏭️  Optimisation ignorée (modèle par défaut conservé)")

# ====================================
# 10. SAUVEGARDE DU MODÈLE
# ====================================

print("\n💾 Sauvegarde du modèle...")
joblib.dump(model, 'models/svm.pkl')
print("✅ Modèle sauvegardé : models/svm.pkl")

# Sauvegarde les métriques
metrics = {
    'Model': 'SVM',
    'Test_Accuracy': accuracy,
    'CV_Mean': mean_cv,
    'CV_Std': std_cv,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'ROC-AUC': roc_auc,
    'Train_Accuracy': train_acc,
    'Overfitting': train_acc - accuracy,
    'N_Support_Vectors': sum(n_support)
}

pd.DataFrame([metrics]).to_csv('models/svm_metrics.csv', index=False)
print("✅ Métriques sauvegardées : models/svm_metrics.csv")

# ====================================
# RÉSUMÉ FINAL
# ====================================

print("\n" + "=" * 60)
print("✅ SVM - ENTRAÎNEMENT TERMINÉ")
print("=" * 60)

print("\n📁 FICHIERS GÉNÉRÉS :")
print("   📂 models/")
print("      ├── svm.pkl")
print("      └── svm_metrics.csv")
print("   📂 images/")
print("      ├── confusion_matrix_svm.png")
print("      └── roc_curve_svm.png")

print(f"\n🎯 RÉSULTATS FINAUX :")
print(f"   Test Accuracy       : {accuracy*100:.2f}%")
print(f"   CV Accuracy (5-fold): {mean_cv*100:.2f}% ± {std_cv*100:.2f}%")
print(f"   Precision           : {precision*100:.2f}%")
print(f"   Recall              : {recall*100:.2f}%")
print(f"   F1-Score            : {f1*100:.2f}%")
print(f"   ROC-AUC             : {roc_auc:.4f}")

