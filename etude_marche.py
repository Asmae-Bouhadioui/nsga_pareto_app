import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
# Configuration de la page en mode large
st.set_page_config(page_title="Dashboard Médical", layout="wide")
# ===============================
# 🔹 Données simulées du marché
# ===============================
market_data = pd.DataFrame({
    "Produit": ["Smartphone", "Laptop", "Casque Audio", "Tablette", "Montre Connectée"],
    "Prix Moyen": [850, 1200, 150, 500, 250],
    "Zone": ["Europe", "USA", "Asie", "Europe", "Afrique"],
    "Catégorie": ["Électronique", "Informatique", "Accessoires", "Électronique", "Wearable"],
    "Ventes": [1200, 800, 1500, 600, 400]
})

st.sidebar.header("Sélectionnez un produit")

# Sélection du produit
produit_selectionne = st.sidebar.selectbox("Choisissez un produit", market_data["Produit"].unique())

# Filtrage des données du produit sélectionné
produit_info = market_data[market_data["Produit"] == produit_selectionne]

# Affichage des informations du produit
st.sidebar.write(f"**📌 Infos Marché pour {produit_selectionne}**")
st.sidebar.write(f"📊 **Prix Moyen :** {produit_info['Prix Moyen'].values[0]} €")
st.sidebar.write(f"📍 **Zone la plus vendue :** {produit_info['Zone'].values[0]}")
st.sidebar.write(f"🏷️ **Catégorie principale :** {produit_info['Catégorie'].values[0]}")
st.sidebar.write(f"📈 **Nombre de ventes :** {produit_info['Ventes'].values[0]} unités")

# ===============================
# 🔹 Interface Utilisateur Streamlit
# ===============================
st.title("📊 Étude de Marché & Optimisation des Prix")

# Création des colonnes pour la mise en page
st.header("Optimisation du prix du produit")
# Colonne 1 : Optimisation du prix et affichage des graphiques
if st.button("Optimiser le prix du produit"):
    # Définition du problème d'optimisation des prix
    class PricingOptimization(Problem):
        def __init__(self, prix_moyen):
            super().__init__(n_var=1, n_obj=3, n_constr=0, xl=np.array([prix_moyen * 0.5]), xu=np.array([prix_moyen * 1.5]))

        def _evaluate(self, X, out, *args, **kwargs):
            price = X[:, 0]
            cost = 0.6 * price  # Supposition : le coût de production est 60% du prix

            # Fonction de demande : plus le prix est élevé, moins il y a d'acheteurs
            demand = np.maximum(0, 10000 - 5 * price)

            # Objectifs à optimiser
            revenue = -(price * demand)  # Maximiser le chiffre d'affaires
            profit_margin = -(price - cost)  # Maximiser la marge bénéficiaire
            customer_satisfaction = np.abs(price - produit_info['Prix Moyen'].values[0])  # Minimiser l'écart avec le prix moyen

            out["F"] = np.column_stack([revenue, customer_satisfaction, profit_margin])

    # Exécution de l'optimisation
    problem = PricingOptimization(prix_moyen=produit_info['Prix Moyen'].values[0])
    algorithm = NSGA2(
        pop_size=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm, termination=get_termination("n_gen", 100), seed=1, verbose=False)

    # Récupération des solutions du Front de Pareto
    F = res.F

    # 🔹 Affichage du Front de Pareto
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=F[:, 0],
        y=F[:, 1],
        mode='markers',
        marker=dict(color=F[:, 2], colorscale='Viridis', size=8, showscale=True),
        name="Solutions optimales"
    ))

    fig.update_layout(
        title=f"Front de Pareto - {produit_selectionne}",
        xaxis_title="Chiffre d'affaires (-)",
        yaxis_title="Insatisfaction client",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

    # ===============================
    # 🔹 Statistiques supplémentaires
    # ===============================
        
    # Création de colonnes pour les graphiques côte à côte
    with st.container():
        col1, col2 = st.columns(2)

        # Graphique des ventes par zone géographique
        with col1:
            fig2 = px.bar(market_data, x="Zone", y="Ventes", color="Produit", title="📍 Ventes par Zone Géographique")
            st.plotly_chart(fig2, use_container_width=True)  # Permet de prendre toute la largeur disponible

        # Graphique des prix moyens par catégorie
        with col2:
            fig3 = px.bar(market_data, x="Catégorie", y="Prix Moyen", color="Produit", title="🏷️ Prix Moyens par Catégorie")
            st.plotly_chart(fig3, use_container_width=True)  # Permet de prendre toute la largeur disponible
