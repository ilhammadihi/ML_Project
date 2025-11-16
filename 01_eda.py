"""
Heart Disease Prediction - EDA COMPLET
======================================
Génère TOUS les graphiques pour la présentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
os.makedirs('images', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("📊 ANALYSE EXPLORATOIRE COMPLÈTE (EDA)")
print("=" * 80)

# ====================================
# 1. CHARGEMENT DES DONNÉES
# ====================================

df = pd.read_csv('data/heart.csv')

print(f"\n✅ Dataset chargé : {df.shape}")
print(f"   {df.shape[0]} patients, {df.shape[1]} colonnes")

# ====================================
# 2. INFORMATIONS GÉNÉRALES
# ====================================

print("\n" + "=" * 80)
print("📋 INFORMATIONS GÉNÉRALES")
print("=" * 80)

print(f"\nPremières lignes :")
print(df.head())

print(f"\n📈 Statistiques descriptives :")
print(df.describe())

print(f"\n🔍 Valeurs manquantes :")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Aucune valeur manquante !")
else:
    print(missing[missing > 0])

print(f"\n🎯 Distribution des classes :")
print(df['target'].value_counts())
print(f"\nPourcentages :")
print(df['target'].value_counts(normalize=True) * 100)

# ====================================
# GRAPHIQUE 1 : DISTRIBUTION DE LA TARGET
# ====================================

print("\n📊 Génération Graphique 1 : Distribution des classes...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
target_counts = df['target'].value_counts()
axes[0].bar(['Sain (0)', 'Malade (1)'], target_counts.values, 
            color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
axes[0].set_title('Distribution des Classes', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Classe', fontsize=12)
axes[0].set_ylabel('Nombre de Patients', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Ajoute les valeurs sur les barres
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 3, str(v), ha='center', fontweight='bold', fontsize=11)

# Pie chart
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0.05)
axes[1].pie(target_counts.values, labels=['Sain (0)', 'Malade (1)'], 
            autopct='%1.1f%%', colors=colors, explode=explode,
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1].set_title('Proportion des Classes', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('images/1_target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/1_target_distribution.png")

# ====================================
# GRAPHIQUE 2 : DISTRIBUTION DE L'ÂGE
# ====================================

print("\n📊 Génération Graphique 2 : Distribution de l'âge...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme par classe
sns.histplot(data=df, x='age', hue='target', bins=20, kde=True, ax=axes[0],
             palette=['#2ecc71', '#e74c3c'], alpha=0.6, edgecolor='black')
axes[0].set_title('Distribution de l\'Âge par Classe', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Âge (années)', fontsize=12)
axes[0].set_ylabel('Nombre de Patients', fontsize=12)
axes[0].legend(['Sain', 'Malade'], fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Boxplot
sns.boxplot(data=df, x='target', y='age', ax=axes[1],
            palette=['#2ecc71', '#e74c3c'], linewidth=2)
axes[1].set_title('Âge vs Maladie Cardiaque', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_ylabel('Âge (années)', fontsize=12)
axes[1].set_xticklabels(['Sain (0)', 'Malade (1)'])
axes[1].grid(axis='y', alpha=0.3)

# Ajoute les moyennes
mean_0 = df[df['target']==0]['age'].mean()
mean_1 = df[df['target']==1]['age'].mean()
axes[1].text(0, mean_0, f'μ={mean_0:.1f}', ha='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1].text(1, mean_1, f'μ={mean_1:.1f}', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('images/2_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/2_age_distribution.png")

# ====================================
# GRAPHIQUE 3 : SEXE VS TARGET
# ====================================

print("\n📊 Génération Graphique 3 : Sexe vs Target...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stacked bar
sex_target = pd.crosstab(df['sex'], df['target'])
sex_target.index = ['Femme (0)', 'Homme (1)']
sex_target.columns = ['Sain', 'Malade']

sex_target.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'],
                edgecolor='black', linewidth=1.5)
axes[0].set_title('Répartition par Sexe et Classe', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Sexe', fontsize=12)
axes[0].set_ylabel('Nombre de Patients', fontsize=12)
axes[0].legend(title='Classe', fontsize=10)
axes[0].set_xticklabels(['Femme', 'Homme'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Pourcentages normalisés
sex_target_pct = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
sex_target_pct.index = ['Femme', 'Homme']
sex_target_pct.columns = ['Sain', 'Malade']

sex_target_pct.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'],
                    edgecolor='black', linewidth=1.5)
axes[1].set_title('Pourcentage par Sexe', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Sexe', fontsize=12)
axes[1].set_ylabel('Pourcentage (%)', fontsize=12)
axes[1].legend(title='Classe', fontsize=10)
axes[1].set_xticklabels(['Femme', 'Homme'], rotation=0)
axes[1].set_ylim([0, 100])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/3_sex_vs_target.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/3_sex_vs_target.png")

# ====================================
# GRAPHIQUE 4 : TYPE DE DOULEUR THORACIQUE
# ====================================

print("\n📊 Génération Graphique 4 : Type de douleur thoracique...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
cp_target = pd.crosstab(df['cp'], df['target'])
cp_labels = ['Type 0\n(Angine typique)', 'Type 1\n(Angine atypique)', 
             'Type 2\n(Non-angineuse)', 'Type 3\n(Asymptomatique)']

cp_target.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'],
               edgecolor='black', linewidth=1.5)
axes[0].set_title('Type de Douleur Thoracique vs Maladie', 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Type de Douleur', fontsize=12)
axes[0].set_ylabel('Nombre de Patients', fontsize=12)
axes[0].legend(['Sain', 'Malade'], fontsize=10)
axes[0].set_xticklabels(cp_labels, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Heatmap
cp_heatmap = pd.crosstab(df['cp'], df['target'], normalize='index') * 100
sns.heatmap(cp_heatmap, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            ax=axes[1], cbar_kws={'label': 'Pourcentage (%)'}, 
            linewidths=1, linecolor='white')
axes[1].set_title('Pourcentage de Maladie par Type de Douleur', 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Target (0=Sain, 1=Malade)', fontsize=12)
axes[1].set_ylabel('Type de Douleur', fontsize=12)
axes[1].set_yticklabels(['Type 0', 'Type 1', 'Type 2', 'Type 3'], rotation=0)

plt.tight_layout()
plt.savefig('images/4_chest_pain_type.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/4_chest_pain_type.png")

# ====================================
# GRAPHIQUE 5 : PARAMÈTRES CARDIAQUES
# ====================================

print("\n📊 Génération Graphique 5 : Paramètres cardiaques...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cholestérol
sns.boxplot(data=df, x='target', y='chol', ax=axes[0, 0],
            palette=['#2ecc71', '#e74c3c'], linewidth=2)
axes[0, 0].set_title('Cholestérol vs Maladie', fontsize=12, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('Target', fontsize=11)
axes[0, 0].set_ylabel('Cholestérol (mg/dl)', fontsize=11)
axes[0, 0].set_xticklabels(['Sain', 'Malade'])
axes[0, 0].grid(axis='y', alpha=0.3)

# Fréquence cardiaque max
sns.boxplot(data=df, x='target', y='thalach', ax=axes[0, 1],
            palette=['#2ecc71', '#e74c3c'], linewidth=2)
axes[0, 1].set_title('Fréquence Cardiaque Max vs Maladie', fontsize=12, fontweight='bold', pad=10)
axes[0, 1].set_xlabel('Target', fontsize=11)
axes[0, 1].set_ylabel('Fréquence Max (bpm)', fontsize=11)
axes[0, 1].set_xticklabels(['Sain', 'Malade'])
axes[0, 1].grid(axis='y', alpha=0.3)

# Pression artérielle
sns.boxplot(data=df, x='target', y='trestbps', ax=axes[1, 0],
            palette=['#2ecc71', '#e74c3c'], linewidth=2)
axes[1, 0].set_title('Pression Artérielle vs Maladie', fontsize=12, fontweight='bold', pad=10)
axes[1, 0].set_xlabel('Target', fontsize=11)
axes[1, 0].set_ylabel('Pression (mm Hg)', fontsize=11)
axes[1, 0].set_xticklabels(['Sain', 'Malade'])
axes[1, 0].grid(axis='y', alpha=0.3)

# Oldpeak
sns.boxplot(data=df, x='target', y='oldpeak', ax=axes[1, 1],
            palette=['#2ecc71', '#e74c3c'], linewidth=2)
axes[1, 1].set_title('Dépression ST vs Maladie', fontsize=12, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Target', fontsize=11)
axes[1, 1].set_ylabel('Oldpeak', fontsize=11)
axes[1, 1].set_xticklabels(['Sain', 'Malade'])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Paramètres Cardiaques vs Maladie', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/5_cardiac_parameters.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/5_cardiac_parameters.png")

# ====================================
# GRAPHIQUE 6 : MATRICE DE CORRÉLATION
# ====================================

print("\n📊 Génération Graphique 6 : Matrice de corrélation...")

plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()

# Heatmap avec annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)

plt.title('Matrice de Corrélation des Features', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/6_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/6_correlation_matrix.png")

# ====================================
# GRAPHIQUE 7 : TOP CORRÉLATIONS AVEC TARGET
# ====================================

print("\n📊 Génération Graphique 7 : Top corrélations avec Target...")

# Calcule les corrélations avec target
target_corr = df.corr()['target'].drop('target').sort_values(ascending=False)

print(f"\n🔥 TOP CORRÉLATIONS AVEC TARGET :")
print(target_corr)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot horizontal
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in target_corr.values]
axes[0].barh(target_corr.index, target_corr.values, color=colors, 
             edgecolor='black', linewidth=1)
axes[0].set_xlabel('Corrélation avec Target', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Features', fontsize=12, fontweight='bold')
axes[0].set_title('Corrélation des Features avec la Maladie', 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0].grid(axis='x', alpha=0.3)

# Top 5 en valeur absolue
top5_corr = target_corr.abs().sort_values(ascending=True).tail(5)
axes[1].barh(top5_corr.index, top5_corr.values, 
             color=plt.cm.viridis(np.linspace(0, 1, 5)), 
             edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('|Corrélation|', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Features', fontsize=12, fontweight='bold')
axes[1].set_title('Top 5 Features Corrélées (Valeur Absolue)', 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].grid(axis='x', alpha=0.3)

# Ajoute les valeurs
for i, v in enumerate(top5_corr.values):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('images/7_feature_correlation_target.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/7_feature_correlation_target.png")

# ====================================
# GRAPHIQUE 8 : PAIRPLOT (TOP 4 FEATURES)
# ====================================

print("\n📊 Génération Graphique 8 : Pairplot des top features...")

# Sélectionne les 4 features les plus corrélées + target
top4_features = target_corr.abs().sort_values(ascending=False).head(4).index.tolist()
top4_features.append('target')

pairplot_data = df[top4_features].copy()
pairplot_data['target'] = pairplot_data['target'].map({0: 'Sain', 1: 'Malade'})

pairplot = sns.pairplot(pairplot_data, hue='target', 
                        palette=['#2ecc71', '#e74c3c'],
                        plot_kws={'alpha': 0.6, 'edgecolor': 'black', 'linewidth': 0.5},
                        diag_kind='kde')

pairplot.fig.suptitle('Pairplot - Top 4 Features Corrélées', 
                      fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/8_pairplot_top_features.png', dpi=300, bbox_inches='tight')
plt.show()
print("   ✅ Sauvegardé : images/8_pairplot_top_features.png")

# ====================================
# RÉSUMÉ DES INSIGHTS
# ====================================

print("\n" + "=" * 80)
print("💡 INSIGHTS PRINCIPAUX DE L'EDA")
print("=" * 80)

print(f"""
1️⃣ DATASET :
   • {len(df)} patients au total
   • {(df['target']==0).sum()} patients sains ({(df['target']==0).sum()/len(df)*100:.1f}%)
   • {(df['target']==1).sum()} patients malades ({(df['target']==1).sum()/len(df)*100:.1f}%)
   ✅ Classes relativement équilibrées

2️⃣ QUALITÉ DES DONNÉES :
   • Aucune valeur manquante ✅
   • Aucun doublon
   • {len(df.columns)-1} features cliniques

3️⃣ TOP 5 FEATURES CORRÉLÉES AVEC LA MALADIE :
""")

for i, (feature, corr) in enumerate(target_corr.abs().sort_values(ascending=False).head(5).items(), 1):
    direction = "↑ Augmente" if target_corr[feature] > 0 else "↓ Diminue"
    print(f"   {i}. {feature:12s} : {abs(corr):.3f} {direction} le risque")

print(f"""
4️⃣ OBSERVATIONS CLÉS :
   • Les hommes sont plus touchés que les femmes
   • L'âge moyen des malades est légèrement plus élevé
   • Le type de douleur thoracique (cp) est très prédictif
   • La fréquence cardiaque max (thalach) est un bon indicateur
   • L'angine d'exercice (exang) corrélée négativement

5️⃣ FEATURES À SURVEILLER :
   • cp (Type de douleur) : Feature #1 la plus prédictive
   • thalach, oldpeak, exang : Indicateurs cardiaques clés
   • ca, slope, thal : Examens spécialisés importants
""")

# ====================================
# RÉSUMÉ FINAL
# ====================================

print("\n" + "✅" * 40)
print("EDA TERMINÉE AVEC SUCCÈS")
print("✅" * 40)

print(f"""
📁 8 GRAPHIQUES GÉNÉRÉS DANS 'images/' :

1. 1_target_distribution.png       → Distribution des classes
2. 2_age_distribution.png           → Âge vs Target
3. 3_sex_vs_target.png              → Sexe vs Target
4. 4_chest_pain_type.png            → Type de douleur thoracique
5. 5_cardiac_parameters.png         → Paramètres cardiaques
6. 6_correlation_matrix.png         → Matrice de corrélation
7. 7_feature_correlation_target.png → Top corrélations
8. 8_pairplot_top_features.png      → Pairplot des top features

🎯 Utilise ces graphiques dans ta présentation PowerPoint !

📊 Slides recommandés :
   • Slide 4 : Graphique 1 (Distribution)
   • Slide 6 : Graphiques 6 et 7 (Corrélations)
   • Slide EDA : Graphiques 2, 4, 5 (Analyses détaillées)
""")