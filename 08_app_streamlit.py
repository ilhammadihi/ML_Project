"""
Heart Disease Prediction - Streamlit Application
================================================
Application web interactive pour prédire les maladies cardiaques
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import sys

# Configuration de la page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #e74c3c;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================
# FONCTIONS UTILITAIRES
# ====================================

@st.cache_resource
def load_models():
    """Charge tous les modèles ML"""
    models = {}
    model_names = ['logistic_regression', 'random_forest', 'random_forest_regularized', 'svm', 'best_model']
    
    for name in model_names:
        try:
            models[name] = joblib.load(f'../models/{name}.pkl')
        except:
            try:
                models[name] = joblib.load(f'models/{name}.pkl')
            except:
                pass
    
    return models

@st.cache_resource
def load_scaler():
    """Charge le scaler"""
    try:
        return joblib.load('../models/scaler.pkl')
    except:
        try:
            return joblib.load('models/scaler.pkl')
        except:
            return None

def get_model_name(filename):
    """Convertit nom de fichier en nom lisible"""
    names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'random_forest_regularized': 'Random Forest (Regularized)',
        'svm': 'Support Vector Machine',
        'best_model': 'Best Model'
    }
    return names.get(filename, filename)

def create_gauge_chart(probability):
    """Crée un graphique jauge pour la probabilité"""
    
    # Couleur selon le risque
    if probability < 0.3:
        color = "green"
        risk_level = "FAIBLE"
    elif probability < 0.7:
        color = "orange"
        risk_level = "MODÉRÉ"
    else:
        color = "red"
        risk_level = "ÉLEVÉ"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Risque de Maladie Cardiaque<br><b>{risk_level}</b>", 
                 'font': {'size': 20}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart(model, feature_names):
    """Crée un graphique d'importance des features"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Importance des Features',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Importance",
            yaxis_title="Features"
        )
        
        return fig
    
    elif hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        }).sort_values('Coefficient', ascending=True)
        
        fig = px.bar(
            feature_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Coefficients du Modèle',
            color='Coefficient',
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Coefficient",
            yaxis_title="Features"
        )
        
        return fig
    
    return None

# ====================================
# HEADER
# ====================================

st.markdown("# 🫀 Heart Disease Prediction")
st.markdown("### Application ML pour la Prédiction de Maladies Cardiaques")
st.markdown("---")

# ====================================
# SIDEBAR - INFORMATIONS
# ====================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=100)
    st.markdown("## 📊 À propos")
    st.info("""
    **Projet End-to-End ML**
    
    Cette application utilise le Machine Learning pour prédire 
    la présence de maladies cardiaques basé sur 13 paramètres cliniques.
    
    **Dataset**: UCI Heart Disease  
    **Patients**: 303  
    **Modèles**: 3 algorithmes comparés
    """)
    
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "Choisir une page:",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Comparaison Modèles", "📈 Statistiques", "ℹ️ Documentation"]
    )
    
    st.markdown("---")
    st.markdown("**Développé par:** [Ton Nom]")
    st.markdown("**Projet:** ML INE2-DATA 2025")

# ====================================
# CHARGEMENT DES MODÈLES
# ====================================

models = load_models()
scaler = load_scaler()

if len(models) == 0:
    st.error("❌ Aucun modèle trouvé ! Assure-toi que les modèles sont dans le dossier 'models/'")
    st.stop()

# ====================================
# PAGE 1 : ACCUEIL
# ====================================

if page == "🏠 Accueil":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Objectif du Projet")
        st.write("""
        Les maladies cardiovasculaires sont la **première cause de mortalité mondiale** (OMS).
        
        Cette application utilise des algorithmes de Machine Learning pour :
        - ✅ Prédire la présence d'une maladie cardiaque
        - ✅ Identifier les facteurs de risque principaux
        - ✅ Aider à la décision médicale précoce
        """)
        
        st.markdown("## 📊 Dataset")
        st.write("""
        **UCI Heart Disease Dataset**
        - 303 patients
        - 13 features cliniques
        - Classes équilibrées (45% sains, 55% malades)
        """)
        
        st.markdown("## 🤖 Modèles Implémentés")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.success("**Logistic Regression**\n\nApproche linéaire classique")
        with col_b:
            st.success("**Random Forest**\n\nEnsemble d'arbres de décision")
        with col_c:
            st.success("**SVM**\n\nSupport Vector Machine")
    
    with col2:
        st.markdown("## 📈 Performances")
        
        # Affiche les métriques du meilleur modèle
        try:
            comparison_df = pd.read_csv('../models/final_comparison.csv')
        except:
            try:
                comparison_df = pd.read_csv('models/final_comparison.csv')
            except:
                comparison_df = None
        
        if comparison_df is not None:
            best_model_row = comparison_df.iloc[comparison_df['Overall_Score'].argmax()]
            
            st.metric("🏆 Meilleur Modèle", best_model_row['Model'])
            st.metric("📊 CV Accuracy", f"{best_model_row['CV_Mean']*100:.1f}%")
            st.metric("🎯 ROC-AUC", f"{best_model_row['ROC-AUC']:.3f}")
        else:
            st.info("Métriques non disponibles")
        
        st.markdown("---")
        st.markdown("## 🚀 Démarrage Rapide")
        st.write("""
        1. Va sur **🔮 Prédiction**
        2. Entre les paramètres cliniques
        3. Obtiens la prédiction instantanée
        """)

# ====================================
# PAGE 2 : PRÉDICTION
# ====================================

elif page == "🔮 Prédiction":
    
    st.markdown("## 🔮 Prédiction de Maladie Cardiaque")
    st.write("Entre les paramètres cliniques du patient pour obtenir une prédiction.")
    
    # Sélection du modèle
    st.markdown("### 1️⃣ Choix du Modèle")
    available_models = {get_model_name(k): k for k in models.keys()}
    selected_model_name = st.selectbox(
        "Sélectionne un modèle:",
        options=list(available_models.keys())
    )
    selected_model_key = available_models[selected_model_name]
    model = models[selected_model_key]
    
    st.markdown("---")
    st.markdown("### 2️⃣ Paramètres du Patient")
    
    # Formulaire de saisie
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Informations Démographiques")
        age = st.slider("Âge", 29, 77, 50, help="Âge du patient en années")
        sex = st.selectbox("Sexe", [1, 0], format_func=lambda x: "Homme" if x == 1 else "Femme")
        
        st.markdown("#### 🩺 Examens de Base")
        trestbps = st.slider("Pression Artérielle au Repos (mm Hg)", 94, 200, 120)
        chol = st.slider("Cholestérol (mg/dl)", 126, 564, 200)
        fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1], 
                          format_func=lambda x: "Oui" if x == 1 else "Non")
    
    with col2:
        st.markdown("#### 💓 Douleur Thoracique")
        cp = st.selectbox("Type de Douleur Thoracique", [0, 1, 2, 3],
                         format_func=lambda x: {
                             0: "Angine typique",
                             1: "Angine atypique", 
                             2: "Douleur non-angineuse",
                             3: "Asymptomatique"
                         }[x])
        
        st.markdown("#### 🫀 Tests Cardiaques")
        restecg = st.selectbox("Résultats ECG au Repos", [0, 1, 2],
                              format_func=lambda x: {
                                  0: "Normal",
                                  1: "Anomalie ST-T",
                                  2: "Hypertrophie"
                              }[x])
        thalach = st.slider("Fréquence Cardiaque Maximale", 71, 202, 150)
        exang = st.selectbox("Angine induite par l'exercice", [0, 1],
                            format_func=lambda x: "Oui" if x == 1 else "Non")
    
    with col3:
        st.markdown("#### 📊 Autres Indicateurs")
        oldpeak = st.slider("Dépression ST induite par l'exercice", 0.0, 6.2, 1.0, 0.1)
        slope = st.selectbox("Pente du segment ST", [0, 1, 2],
                            format_func=lambda x: {
                                0: "Montante",
                                1: "Plate",
                                2: "Descendante"
                            }[x])
        ca = st.selectbox("Nombre de vaisseaux colorés", [0, 1, 2, 3])
        thal = st.selectbox("Thalassémie", [1, 2, 3],
                           format_func=lambda x: {
                               1: "Normal",
                               2: "Défaut fixe",
                               3: "Défaut réversible"
                           }[x])
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("🔮 PRÉDIRE", type="primary", use_container_width=True):
        
        # Prépare les données
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        # Scaling
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
            input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)
        else:
            input_scaled = input_data
        
        # Prédiction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.markdown("### 3️⃣ Résultats de la Prédiction")
        
        # Affichage du résultat
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            if prediction == 1:
                st.error("### ⚠️ MALADIE DÉTECTÉE")
                st.write("Le modèle prédit la **présence** d'une maladie cardiaque.")
            else:
                st.success("### ✅ PAS DE MALADIE")
                st.write("Le modèle prédit l'**absence** de maladie cardiaque.")
            
            st.metric("Probabilité Maladie", f"{probability[1]*100:.1f}%")
            st.metric("Probabilité Sain", f"{probability[0]*100:.1f}%")
        
        with col_res2:
            # Graphique Jauge
            gauge_fig = create_gauge_chart(probability[1])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Disclaimer médical
        st.warning("""
        ⚠️ **Disclaimer Médical**
        
        Cette prédiction est fournie à titre informatif uniquement et ne remplace pas 
        un diagnostic médical professionnel. Consultez toujours un médecin qualifié 
        pour toute question de santé.
        """)
        
        # Feature Importance
        st.markdown("---")
        st.markdown("### 📊 Facteurs Influençant la Prédiction")
        
        importance_fig = create_feature_importance_chart(model, input_data.columns.tolist())
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)

# ====================================
# PAGE 3 : COMPARAISON MODÈLES
# ====================================

elif page == "📊 Comparaison Modèles":
    
    st.markdown("## 📊 Comparaison des Modèles ML")
    
    try:
        comparison_df = pd.read_csv('../models/final_comparison.csv')
    except:
        try:
            comparison_df = pd.read_csv('models/final_comparison.csv')
        except:
            st.error("❌ Fichier de comparaison non trouvé!")
            st.stop()
    
    # Tableau comparatif
    st.markdown("### 📋 Tableau Comparatif des Performances")
    
    display_df = comparison_df[['Model', 'Test_Acc', 'CV_Mean', 'CV_Std', 
                                 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].copy()
    
    # Formate en pourcentages
    for col in ['Test_Acc', 'CV_Mean', 'CV_Std', 'Precision', 'Recall', 'F1-Score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    display_df['ROC-AUC'] = display_df['ROC-AUC'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Meilleur modèle
    best_idx = comparison_df['Overall_Score'].argmax()
    best_model = comparison_df.iloc[best_idx]
    
    st.success(f"""
    🏆 **Meilleur Modèle : {best_model['Model']}**
    
    - CV Accuracy : {best_model['CV_Mean']*100:.2f}% ± {best_model['CV_Std']*100:.2f}%
    - ROC-AUC : {best_model['ROC-AUC']:.4f}
    - F1-Score : {best_model['F1-Score']*100:.2f}%
    """)
    
    st.markdown("---")
    
    # Graphiques
    st.markdown("### 📈 Visualisations")
    
    tab1, tab2, tab3 = st.tabs(["📊 Métriques", "📈 ROC Curves", "🎯 Overfitting"])
    
    with tab1:
        # Barres comparatives
        metrics_df = comparison_df[['Model', 'Test_Acc', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].set_index('Model')
        
        fig = px.bar(
            metrics_df.T,
            barmode='group',
            title='Comparaison des Métriques',
            labels={'value': 'Score', 'index': 'Métrique'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.info("📈 Les courbes ROC sont disponibles dans le dossier images/roc_curves_all.png")
        try:
            st.image("../images/roc_curves_all.png")
        except:
            try:
                st.image("images/roc_curves_all.png")
            except:
                st.warning("Image ROC non trouvée")
    
    with tab3:
        # Train vs Test
        overfitting_df = comparison_df[['Model', 'Train_Acc', 'Test_Acc', 'Overfitting']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Train', x=overfitting_df['Model'], y=overfitting_df['Train_Acc']))
        fig.add_trace(go.Bar(name='Test', x=overfitting_df['Model'], y=overfitting_df['Test_Acc']))
        
        fig.update_layout(
            title='Train vs Test Accuracy (Détection Overfitting)',
            xaxis_title='Modèle',
            yaxis_title='Accuracy',
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# ====================================
# PAGE 4 : STATISTIQUES
# ====================================

elif page == "📈 Statistiques":
    
    st.markdown("## 📈 Statistiques du Dataset")
    
    # Charge le dataset
    try:
        df = pd.read_csv('../data/heart.csv')
    except:
        try:
            df = pd.read_csv('data/heart.csv')
        except:
            st.error("❌ Dataset non trouvé!")
            st.stop()
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Patients", len(df))
    with col2:
        st.metric("✅ Patients Sains", (df['target']==0).sum())
    with col3:
        st.metric("❌ Patients Malades", (df['target']==1).sum())
    with col4:
        st.metric("📋 Features", len(df.columns)-1)
    
    st.markdown("---")
    
    # Visualisations
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔥 Corrélations", "📈 Analyses"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Distribution target
            fig = px.pie(
                df, 
                names='target',
                title='Distribution des Classes',
                color='target',
                color_discrete_map={0: 'green', 1: 'red'}
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            # Distribution âge
            fig = px.histogram(
                df,
                x='age',
                color='target',
                title='Distribution de l\'Âge',
                nbins=20,
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Heatmap corrélations
        corr_matrix = df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Matrice de Corrélation',
            color_continuous_scale='RdBu'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Boxplots par feature
        selected_feature = st.selectbox(
            "Sélectionne une feature:",
            options=[col for col in df.columns if col != 'target']
        )
        
        fig = px.box(
            df,
            x='target',
            y=selected_feature,
            color='target',
            title=f'{selected_feature} vs Target',
            color_discrete_map={0: 'green', 1: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tableau de données
    with st.expander("📋 Voir le Dataset Complet"):
        st.dataframe(df, use_container_width=True)

# ====================================
# PAGE 5 : DOCUMENTATION
# ====================================

elif page == "ℹ️ Documentation":
    
    st.markdown("## ℹ️ Documentation du Projet")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Guide", "🔬 Features", "🤖 Modèles", "📚 Références"])
    
    with tab1:
        st.markdown("""
        ### 📖 Guide d'Utilisation
        
        #### 1. Navigation
        Utilise la barre latérale pour naviguer entre les différentes pages.
        
        #### 2. Faire une Prédiction
        - Va sur la page **🔮 Prédiction**
        - Sélectionne un modèle
        - Entre les 13 paramètres cliniques
        - Clique sur **PRÉDIRE**
        - Analyse les résultats et la probabilité
        
        #### 3. Comparer les Modèles
        - Va sur **📊 Comparaison Modèles**
        - Consulte le tableau comparatif
        - Explore les visualisations
        
        #### 4. Explorer les Données
        - Va sur **📈 Statistiques**
        - Analyse les distributions
        - Consulte les corrélations
        """)
    
    with tab2:
        st.markdown("""
        ### 🔬 Description des Features
        
        | Feature | Description | Unité | Valeurs |
        |---------|-------------|-------|---------|
        | **age** | Âge du patient | années | 29-77 |
        | **sex** | Sexe | - | 0=F, 1=M |
        | **cp** | Type de douleur thoracique | - | 0-3 |
        | **trestbps** | Pression artérielle au repos | mm Hg | 94-200 |
        | **chol** | Cholestérol sérique | mg/dl | 126-564 |
        | **fbs** | Glycémie à jeun > 120 mg/dl | - | 0=Non, 1=Oui |
        | **restecg** | Résultats ECG au repos | - | 0-2 |
        | **thalach** | Fréquence cardiaque max | bpm | 71-202 |
        | **exang** | Angine induite par l'exercice | - | 0=Non, 1=Oui |
        | **oldpeak** | Dépression ST | - | 0-6.2 |
        | **slope** | Pente du segment ST | - | 0-2 |
        | **ca** | Nombre de vaisseaux colorés | - | 0-3 |
        | **thal** | Thalassémie | - | 1-3 |
        
        ### 🎯 Variable Cible
        - **target** : 0 = Pas de maladie, 1 = Maladie présente
        """)
    
    with tab3:
        st.markdown("""
        ### 🤖 Modèles Implémentés
        
        #### 1. Logistic Regression
        - **Type** : Modèle linéaire
        - **Avantages** : Simple, interprétable, rapide
        - **Inconvénients** : Assume linéarité
        - **Usage** : Baseline, interprétation des coefficients
        
        #### 2. Random Forest
        - **Type** : Ensemble d'arbres de décision
        - **Avantages** : Robuste, feature importance, non-linéaire
        - **Inconvénients** : Moins interprétable
        - **Usage** : Haute performance, gestion complexité
        
        #### 3. Support Vector Machine (SVM)
        - **Type** : Maximisation de marge
        - **Avantages** : Kernel trick, robuste, performant
        - **Inconvénients** : Temps de calcul, hyperparamètres
        - **Usage** : Meilleure performance globale
        
        ### 📊 Métriques d'Évaluation
        - **Accuracy** : Taux de prédictions correctes
        - **Precision** : Vrais positifs / (Vrais + Faux positifs)
        - **Recall** : Vrais positifs / (Vrais positifs + Faux négatifs)
        - **F1-Score** : Moyenne harmonique Precision/Recall
        - **ROC-AUC** : Aire sous la courbe ROC
        """)
    
    with tab4:
        st.markdown("""
        ### 📚 Références
        
        #### Dataset
        - **Source** : UCI Machine Learning Repository
        - **Lien** : [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
        
        #### Technologies
        - **Python** : 3.9+
        - **Scikit-learn** : 1.3.0
        - **Streamlit** : 1.28.0
        - **Pandas** : 2.0.3
        - **Plotly** : 5.17.0
        
        #### Bibliographie
        - Aurélien Géron - *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
        - Scikit-learn Documentation
        - Streamlit Documentation
        
        #### Auteur
        - **Projet** : End-to-End ML Project
        - **Cours** : Machine Learning INE2-DATA 2025
        - **Date** : Novembre 2024
        
        ---
        
        ### 📞 Contact
        Pour toute question ou suggestion :
        - GitHub : [Ton repo]
        - Email : [Ton email]
        - LinkedIn : [Ton profil]
        """)

# ====================================
# FOOTER
# ====================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🫀 Heart Disease Prediction | Machine Learning Project 2024</p>
    <p>Développé avec ❤️ pour la santé publique</p>
</div>
""", unsafe_allow_html=True)