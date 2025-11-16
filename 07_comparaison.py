"""
Heart Disease Prediction - Comparaison Finale
=============================================
Compare Logistic Regression, Random Forest et SVM
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ CRÉÉ LES DOSSIERS
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("\n" + "🔬" * 40)
print("COMPARAISON FINALE DES 3 MODÈLES")
print("🔬" * 40)

# ====================================
# 1. CHARGEMENT DES DONNÉES
# ====================================

print("\n📥 Chargement des données...")
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Dataset complet pour CV
X_full = pd.concat([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

print(f"✅ Train : {X_train.shape}")
print(f"✅ Test  : {X_test.shape}")
print(f"✅ Full  : {X_full.shape}")

# ====================================
# 2. CHARGEMENT DES MODÈLES
# ====================================

print("\n📦 Chargement des modèles...")
models = {}

# Logistic Regression
try:
    models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
    print("   ✅ Logistic Regression")
except:
    print("   ❌ Logistic Regression non trouvé")

# Random Forest (version régularisée si existe, sinon version normale)
try:
    models['Random Forest'] = joblib.load('models/random_forest_regularized.pkl')
    print("   ✅ Random Forest (Régularisé)")
except:
    try:
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        print("   ✅ Random Forest")
    except:
        print("   ❌ Random Forest non trouvé")

# SVM
try:
    models['SVM'] = joblib.load('models/svm.pkl')
    print("   ✅ SVM")
except:
    print("   ❌ SVM non trouvé")

if len(models) == 0:
    raise Exception("\n❌ Aucun modèle trouvé ! Lance d'abord les scripts d'entraînement.")

print(f"\n📊 {len(models)} modèle(s) chargé(s)")

# ====================================
# 3. ÉVALUATION COMPLÈTE
# ====================================

print("\n" + "=" * 80)
print("📊 ÉVALUATION COMPLÈTE DES MODÈLES")
print("=" * 80)

results = []
predictions = {}

for name, model in models.items():
    print(f"\n🔍 Évaluation de {name}...")
    
    # ---- TRAIN SET ----
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # ---- TEST SET ----
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # ---- CROSS-VALIDATION ----
    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # ---- OVERFITTING CHECK ----
    overfitting = train_acc - test_acc
    
    # Sauvegarde pour visualisations
    predictions[name] = {
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    }
    
    # Résultats
    metrics = {
        'Model': name,
        'Train_Acc': train_acc,
        'Test_Acc': test_acc,
        'CV_Mean': cv_mean,
        'CV_Std': cv_std,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Overfitting': overfitting
    }
    
    results.append(metrics)
    
    # Affichage
    print(f"   Train Acc    : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Acc     : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   CV Acc       : {cv_mean:.4f} ± {cv_std:.4f} ({cv_mean*100:.2f}%)")
    print(f"   Precision    : {precision:.4f}")
    print(f"   Recall       : {recall:.4f}")
    print(f"   F1-Score     : {f1:.4f}")
    print(f"   ROC-AUC      : {roc_auc:.4f}")
    print(f"   Overfitting  : {overfitting:.4f} ({overfitting*100:.2f}%)")

# ====================================
# 4. TABLEAU DE COMPARAISON
# ====================================

results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')

print("\n" + "=" * 100)
print("📊 TABLEAU COMPARATIF COMPLET")
print("=" * 100)

# Affichage formaté
display_df = results_df.copy()
for col in display_df.columns:
    if col != 'Model':
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

print("\n" + display_df.to_string())

# ====================================
# 5. IDENTIFICATION DU MEILLEUR MODÈLE
# ====================================

print("\n" + "=" * 100)
print("🏆 ANALYSE DES MEILLEURS SCORES")
print("=" * 100)

# Meilleurs par métrique
print("\n📊 Meilleur modèle par métrique :")
metrics_to_compare = ['Test_Acc', 'CV_Mean', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for metric in metrics_to_compare:
    best_model = results_df[metric].idxmax()
    best_score = results_df[metric].max()
    print(f"   {metric:12s} : {best_model:25s} ({best_score:.4f})")

# Score global (moyenne des métriques principales)
results_df['Overall_Score'] = results_df[['CV_Mean', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].mean(axis=1)
best_overall = results_df['Overall_Score'].idxmax()
best_score = results_df['Overall_Score'].max()

print("\n" + "🏆" * 50)
print(f"MEILLEUR MODÈLE GLOBAL : {best_overall}")
print(f"Score global : {best_score:.4f} ({best_score*100:.2f}%)")
print("🏆" * 50)

# Détails du meilleur
print(f"\n📊 Détails de {best_overall} :")
best_row = results_df.loc[best_overall]
print(f"   Test Accuracy       : {best_row['Test_Acc']:.4f} ({best_row['Test_Acc']*100:.2f}%)")
print(f"   CV Accuracy (5-fold): {best_row['CV_Mean']:.4f} ± {best_row['CV_Std']:.4f}")
print(f"   Precision           : {best_row['Precision']:.4f}")
print(f"   Recall              : {best_row['Recall']:.4f}")
print(f"   F1-Score            : {best_row['F1-Score']:.4f}")
print(f"   ROC-AUC             : {best_row['ROC-AUC']:.4f}")

# ====================================
# 6. VISUALISATIONS
# ====================================

print("\n" + "📈" * 40)
print("GÉNÉRATION DES VISUALISATIONS")
print("📈" * 40)

# ---- 6.1 : Comparaison Train vs Test (Overfitting) ----
print("\n📊 Graphique 1 : Train vs Test Accuracy...")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

train_accs = results_df['Train_Acc'].values
test_accs = results_df['Test_Acc'].values

bars1 = ax.bar(x - width/2, train_accs, width, label='Train', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='coral', edgecolor='black')

ax.set_xlabel('Modèles', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Train vs Test Accuracy (Détection Overfitting)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=15, ha='right')
ax.set_ylim([0.75, 1.05])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Valeurs sur barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('images/train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ images/train_vs_test_comparison.png")

# ---- 6.2 : Comparaison CV Scores ----
print("\n📊 Graphique 2 : CV Accuracy avec intervalles de confiance...")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))

cv_means = results_df['CV_Mean'].values
cv_stds = results_df['CV_Std'].values

bars = ax.bar(x, cv_means, color=['#3498db', '#2ecc71', '#e74c3c'], 
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax.errorbar(x, cv_means, yerr=cv_stds, fmt='none', color='black', 
            capsize=5, capthick=2, elinewidth=2)

ax.set_xlabel('Modèles', fontsize=12, fontweight='bold')
ax.set_ylabel('CV Accuracy (5-Fold)', fontsize=12, fontweight='bold')
ax.set_title('Validation Croisée avec Intervalles de Confiance', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=15, ha='right')
ax.set_ylim([0.75, 1.0])
ax.grid(axis='y', alpha=0.3)

# Valeurs
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('images/cv_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ images/cv_accuracy_comparison.png")

# ---- 6.3 : Toutes les métriques (barres groupées) ----
print("\n📊 Graphique 3 : Toutes les métriques...")

metrics_plot = results_df[['Test_Acc', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].copy()

fig, ax = plt.subplots(figsize=(14, 7))
metrics_plot.T.plot(kind='bar', ax=ax, width=0.8, 
                    color=['#3498db', '#2ecc71', '#e74c3c'],
                    edgecolor='black', linewidth=1)

ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comparaison Complète des Performances', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticklabels(metrics_plot.columns, rotation=45, ha='right')
ax.set_ylim([0.7, 1.0])
ax.legend(title='Modèles', loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ images/all_metrics_comparison.png")

# ---- 6.4 : Courbes ROC comparées ----
print("\n📊 Graphique 4 : Courbes ROC...")

plt.figure(figsize=(10, 8))

colors = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71', 'SVM': '#e74c3c'}

for name, preds in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
    auc = roc_auc_score(y_test, preds['y_proba'])
    
    color = colors.get(name, 'black')
    plt.plot(fpr, tpr, color=color, lw=3, 
             label=f'{name} (AUC = {auc:.4f})')

# Ligne diagonale
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Classificateur Aléatoire', alpha=0.5)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12, fontweight='bold')
plt.title('Comparaison des Courbes ROC', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/roc_curves_all.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ images/roc_curves_all.png")

# ---- 6.5 : Heatmap ----
print("\n📊 Graphique 5 : Heatmap des performances...")

plt.figure(figsize=(12, 5))
heatmap_data = results_df[['Test_Acc', 'CV_Mean', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', 
            cbar_kws={'label': 'Score'}, linewidths=2, linecolor='white',
            vmin=0.75, vmax=1.0)

plt.title('Heatmap des Performances', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Métriques', fontsize=12, fontweight='bold')
plt.ylabel('Modèles', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('images/heatmap_all.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ images/heatmap_all.png")

# ---- 6.6 : Radar Chart ----
print("\n📊 Graphique 6 : Radar Chart...")

from math import pi

categories = ['Test Acc', 'CV Mean', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

colors_radar = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71', 'SVM': '#e74c3c'}

for model_name in results_df.index:
    values = results_df.loc[model_name, ['Test_Acc', 'CV_Mean', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].tolist()
    values += values[:1]
    
    color = colors_radar.get(model_name, 'black')
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0.75, 1.0)
ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'])
ax.grid(True)

plt.title('Radar Chart - Comparaison Globale', 
          fontsize=14, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig('images/radar_chart_all.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ images/radar_chart_all.png")

# ====================================
# 7. SAUVEGARDE DES RÉSULTATS
# ====================================

print("\n💾 Sauvegarde des résultats...")

# CSV
results_df.to_csv('models/final_comparison.csv')
print("   ✅ models/final_comparison.csv")

# Rapport texte détaillé
with open('models/final_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("RAPPORT FINAL DE COMPARAISON DES MODÈLES\n")
    f.write("Heart Disease Prediction - End-to-End ML Project\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("MODÈLES COMPARÉS:\n")
    f.write("-" * 100 + "\n")
    for i, model in enumerate(results_df.index, 1):
        f.write(f"{i}. {model}\n")
    f.write("\n")
    
    f.write("TABLEAU COMPLET DES PERFORMANCES:\n")
    f.write("-" * 100 + "\n")
    f.write(results_df.to_string())
    f.write("\n\n")
    
    f.write("MEILLEUR MODÈLE:\n")
    f.write("-" * 100 + "\n")
    f.write(f"Modèle : {best_overall}\n")
    f.write(f"Score global : {best_score:.4f} ({best_score*100:.2f}%)\n\n")
    
    f.write("DÉTAILS DES PERFORMANCES:\n")
    f.write("-" * 100 + "\n")
    for metric, value in results_df.loc[best_overall].items():
        f.write(f"{metric:20s} : {value:.4f}\n")
    
    f.write("\n")
    f.write("CLASSEMENT:\n")
    f.write("-" * 100 + "\n")
    ranking = results_df.sort_values('Overall_Score', ascending=False)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        f.write(f"{medal} {i}. {model:25s} - Score: {row['Overall_Score']:.4f}\n")

print("   ✅ models/final_report.txt")

# Sauvegarde le meilleur modèle
best_model_object = models[best_overall]
joblib.dump(best_model_object, 'models/best_model.pkl')
print(f"   ✅ models/best_model.pkl ({best_overall})")

# ====================================
# 8. RÉSUMÉ FINAL
# ====================================

print("\n" + "✅" * 50)
print("COMPARAISON TERMINÉE AVEC SUCCÈS")
print("✅" * 50)

print("\n📁 FICHIERS GÉNÉRÉS :")
print("   📂 models/")
print("      ├── final_comparison.csv")
print("      ├── final_report.txt")
print("      └── best_model.pkl")
print("\n   📂 images/")
print("      ├── train_vs_test_comparison.png")
print("      ├── cv_accuracy_comparison.png")
print("      ├── all_metrics_comparison.png")
print("      ├── roc_curves_all.png")
print("      ├── heatmap_all.png")
print("      └── radar_chart_all.png")

print(f"\n🏆 CLASSEMENT FINAL :")
ranking = results_df.sort_values('Overall_Score', ascending=False)
for i, (model, row) in enumerate(ranking.iterrows(), 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
    print(f"   {medal} {i}. {model:25s} - Score: {row['Overall_Score']:.4f} ({row['Overall_Score']*100:.2f}%)")

print(f"\n🎯 RECOMMANDATION : {best_overall}")
print(f"   CV Accuracy : {results_df.loc[best_overall, 'CV_Mean']*100:.2f}% ± {results_df.loc[best_overall, 'CV_Std']*100:.2f}%")
print(f"   ROC-AUC     : {results_df.loc[best_overall, 'ROC-AUC']:.4f}")

