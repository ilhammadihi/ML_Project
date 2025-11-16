"""
Diagnostic Complet - Détection Overfitting
==========================================
Analyse approfondie des performances des modèles
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('images', exist_ok=True)

print("=" * 60)
print("🔍 DIAGNOSTIC COMPLET - OVERFITTING")
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
# 2. CHARGEMENT DES MODÈLES
# ====================================

print("\n📦 Chargement des modèles...")
models = {}

try:
    models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
    print("   ✅ Logistic Regression")
except:
    pass

try:
    models['Random Forest'] = joblib.load('models/random_forest.pkl')
    print("   ✅ Random Forest")
except:
    pass

try:
    models['XGBoost'] = joblib.load('models/xgboost.pkl')
    print("   ✅ XGBoost")
except:
    pass

# ====================================
# 3. TEST TRAIN VS TEST
# ====================================

print("\n" + "=" * 60)
print("📊 TEST 1 : COMPARAISON TRAIN VS TEST")
print("=" * 60)

results_train_test = []

for name, model in models.items():
    # Prédictions train
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Prédictions test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Différence
    diff = train_acc - test_acc
    
    results_train_test.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Difference': diff,
        'Overfitting?': 'OUI ⚠️' if diff > 0.1 else 'NON ✅'
    })
    
    print(f"\n🔍 {name}:")
    print(f"   Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Différence     : {diff:.4f} ({diff*100:.2f}%)")
    
    if diff > 0.15:
        print(f"   ⚠️  OVERFITTING SÉVÈRE (diff > 15%)")
    elif diff > 0.1:
        print(f"   ⚠️  OVERFITTING MODÉRÉ (diff > 10%)")
    elif diff > 0.05:
        print(f"   ⚠️  Léger overfitting (diff > 5%)")
    else:
        print(f"   ✅ Pas d'overfitting significatif")
    
    # Confusion Matrix Train
    cm_train = confusion_matrix(y_train, y_train_pred)
    print(f"   Erreurs Train : {(cm_train[0,1] + cm_train[1,0])} / {len(y_train)}")
    
    # Confusion Matrix Test
    cm_test = confusion_matrix(y_test, y_test_pred)
    print(f"   Erreurs Test  : {(cm_test[0,1] + cm_test[1,0])} / {len(y_test)}")

# Tableau récapitulatif
df_train_test = pd.DataFrame(results_train_test)
print("\n📊 TABLEAU RÉCAPITULATIF :")
print(df_train_test.to_string(index=False))

# ====================================
# 4. VALIDATION CROISÉE (K-FOLD)
# ====================================

print("\n" + "=" * 60)
print("📊 TEST 2 : VALIDATION CROISÉE (5-FOLD)")
print("=" * 60)

# Combine train + test pour une vraie CV
X_full = pd.concat([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

print(f"Dataset complet : {X_full.shape}")

cv_results = []

for name, model in models.items():
    print(f"\n🔍 {name}:")
    
    # Cross-validation avec 5 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_full, y_full, cv=kfold, scoring='accuracy')
    
    print(f"   Scores par fold :")
    for i, score in enumerate(scores, 1):
        print(f"      Fold {i} : {score:.4f} ({score*100:.2f}%)")
    
    mean_score = scores.mean()
    std_score = scores.std()
    
    print(f"   Moyenne : {mean_score:.4f} ({mean_score*100:.2f}%)")
    print(f"   Écart-type : {std_score:.4f}")
    
    cv_results.append({
        'Model': name,
        'CV Mean': mean_score,
        'CV Std': std_score,
        'Variance': 'HAUTE ⚠️' if std_score > 0.05 else 'BASSE ✅'
    })
    
    if std_score > 0.1:
        print(f"   ⚠️  VARIANCE TRÈS ÉLEVÉE - Modèle instable !")
    elif std_score > 0.05:
        print(f"   ⚠️  Variance élevée - Possible overfitting")
    else:
        print(f"   ✅ Variance acceptable")

# Tableau CV
df_cv = pd.DataFrame(cv_results)
print("\n📊 TABLEAU VALIDATION CROISÉE :")
print(df_cv.to_string(index=False))

# ====================================
# 5. TEST SUR NOUVEAUX SPLITS
# ====================================

print("\n" + "=" * 60)
print("📊 TEST 3 : STABILITÉ SUR DIFFÉRENTS SPLITS")
print("=" * 60)

from sklearn.model_selection import train_test_split

print("Test sur 5 splits aléatoires différents...")

stability_results = {name: [] for name in models.keys()}

for i in range(5):
    # Nouveau split aléatoire
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.2, random_state=i
    )
    
    for name, model in models.items():
        # Réentraîne sur le nouveau split
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_tr, y_tr)
        
        # Test
        score = accuracy_score(y_te, model_copy.predict(X_te))
        stability_results[name].append(score)

print("\n📊 Résultats sur 5 splits différents :")
for name, scores in stability_results.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\n🔍 {name}:")
    print(f"   Scores : {[f'{s:.3f}' for s in scores]}")
    print(f"   Moyenne : {mean_score:.4f} ({mean_score*100:.2f}%)")
    print(f"   Écart-type : {std_score:.4f}")
    
    if std_score > 0.05:
        print(f"   ⚠️  INSTABLE - Performances varient beaucoup !")
    else:
        print(f"   ✅ Stable")

# ====================================
# 6. VISUALISATIONS
# ====================================

print("\n📈 Génération des visualisations...")

# 6.1 Train vs Test
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

train_accs = [r['Train Accuracy'] for r in results_train_test]
test_accs = [r['Test Accuracy'] for r in results_train_test]

bars1 = ax.bar(x - width/2, train_accs, width, label='Train', color='skyblue')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='coral')

ax.set_xlabel('Modèles', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train vs Test Accuracy (Détection Overfitting)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models.keys(), rotation=15, ha='right')
ax.legend()
ax.set_ylim([0.7, 1.05])
ax.grid(axis='y', alpha=0.3)

# Ajoute les valeurs
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('images/overfitting_diagnosis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Graphique sauvegardé : images/overfitting_diagnosis.png")

# 6.2 Boxplot CV scores
fig, ax = plt.subplots(figsize=(10, 6))

cv_data = []
labels = []

for name, model in models.items():
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_full, y_full, cv=kfold, scoring='accuracy')
    cv_data.append(scores)
    labels.append(name)

ax.boxplot(cv_data, labels=labels)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Distribution des Scores CV (5-Fold)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig('images/cv_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Graphique sauvegardé : images/cv_distribution.png")

# ====================================
# 7. RECOMMANDATIONS
# ====================================

print("\n" + "=" * 60)
print("💡 DIAGNOSTIC FINAL & RECOMMANDATIONS")
print("=" * 60)

# Analyse Random Forest spécifiquement
if 'Random Forest' in models:
    rf_train_test = [r for r in results_train_test if r['Model'] == 'Random Forest'][0]
    
    print("\n🌲 ANALYSE RANDOM FOREST :")
    print(f"   Train Accuracy : {rf_train_test['Train Accuracy']*100:.2f}%")
    print(f"   Test Accuracy  : {rf_train_test['Test Accuracy']*100:.2f}%")
    print(f"   Différence     : {rf_train_test['Difference']*100:.2f}%")
    
    if rf_train_test['Test Accuracy'] >= 0.95:
        print("\n⚠️  PROBLÈME DÉTECTÉ : Accuracy >= 95% sur test")
        print("\n💡 EXPLICATIONS POSSIBLES :")
        print("   1. Dataset très petit (303 patients)")
        print("   2. Features très prédictives (le problème est facile)")
        print("   3. Possible data leakage (vérifier preprocessing)")
        print("   4. Chance statistique (test set trop petit)")
        
        print("\n✅ SOLUTIONS RECOMMANDÉES :")
        print("   1. Utiliser la Validation Croisée (plus fiable)")
        print("   2. Limiter la complexité du modèle :")
        print("      - Réduire max_depth (ex: 3, 5, 7)")
        print("      - Augmenter min_samples_split (ex: 10, 20)")
        print("      - Réduire n_estimators (ex: 50)")
        print("   3. Dans ta présentation :")
        print("      - Mentionne le CV score (plus réaliste)")
        print("      - Explique que c'est un petit dataset")
        print("      - Parle de la simplicité du problème")

print("\n🎯 CONCLUSION :")
print("=" * 60)

# Trouve le meilleur selon CV
best_cv_model = max(cv_results, key=lambda x: x['CV Mean'])
print(f"Meilleur modèle selon CV : {best_cv_model['Model']}")
print(f"Score CV moyen : {best_cv_model['CV Mean']*100:.2f}%")

print("\n💡 POUR TON PROJET :")
print("   - Utilise les scores de Validation Croisée (plus crédibles)")
print("   - Mentionne les limitations du dataset (petit)")
print("   - Explique pourquoi les performances sont élevées")
print("   - Propose des améliorations futures (plus de données)")

print("\n✅ Diagnostic terminé !")