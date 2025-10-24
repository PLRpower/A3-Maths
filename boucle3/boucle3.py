"""
Analyse de Performances Serveur - Stage OPTIMAL
Auteur: Emma
Description: Analyse complète des métriques CPU, Mémoire, Réseau et Température
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("🖥️  ANALYSE DE PERFORMANCES SERVEUR - OPTIMAL")
print("=" * 80)
print()

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================
print("📂 Étape 1: Chargement des données...")

df = pd.read_csv('server_usage_data.csv')
df = df.head(1440)

print(f"✅ Données chargées: {len(df)} mesures sur 24h")
print(f"   Période: {df['Time'].min()} → {df['Time'].max()}")
print()

# ============================================================================
# 2. STATISTIQUES DESCRIPTIVES ET ROBUSTES
# ============================================================================
print("=" * 80)
print("📊 Étape 2: STATISTIQUES DESCRIPTIVES ET ROBUSTES")
print("=" * 80)

metrics = ['CPU_Usage', 'Memory_Usage', 'Network_Usage', 'Temperature']
stats_results = pd.DataFrame()

for metric in metrics:
    values = df[metric]
    stats_results[metric] = {
        'Moyenne': values.mean(),
        'Médiane': values.median(),
        'Écart-type': values.std(),
        'Min': values.min(),
        'Max': values.max(),
        'Q1 (25%)': values.quantile(0.25),
        'Q3 (75%)': values.quantile(0.75),
        'IQR': values.quantile(0.75) - values.quantile(0.25),
        'CV (%)': (values.std() / values.mean()) * 100  # Coefficient de variation
    }

print("\n📈 Tableau des Statistiques:\n")
print(stats_results.round(2).T)
print("\n💡 Interprétation:")
print("   - IQR (Interquartile Range): Mesure robuste de la dispersion")
print("   - CV (Coefficient de Variation): Dispersion relative (en %)")
print()

# ============================================================================
# 3. VISUALISATION DES DONNÉES
# ============================================================================
print("=" * 80)
print("📉 Étape 3: VISUALISATION DES MÉTRIQUES")
print("=" * 80)

fig, axes = plt.subplots(4, 1, figsize=(15, 12))
fig.suptitle('Évolution des Métriques sur 24h - Serveurs OPTIMAL',
             fontsize=16, fontweight='bold', y=0.995)

colors = ['#3b82f6', '#10b981', '#8b5cf6', '#ef4444']
titles = ['CPU Usage (%)', 'Memory Usage (Gb)', 'Network Usage (Mb/s)', 'Temperature (°C)']
thresholds = [80, 12, 150, 70]

for i, (metric, color, title, threshold) in enumerate(zip(metrics, colors, titles, thresholds)):
    ax = axes[i]

    # Ligne principale
    ax.plot(df.index, df[metric], color=color, linewidth=1.5, alpha=0.8)

    # Moyenne
    mean_val = df[metric].mean()
    ax.axhline(mean_val, color=color, linestyle='--', linewidth=2,
               label=f'Moyenne: {mean_val:.2f}', alpha=0.7)

    # Seuil de vigilance
    ax.axhline(threshold, color='red', linestyle=':', linewidth=2,
               label=f'Seuil vigilance: {threshold}', alpha=0.7)

    # Zone de danger
    ax.fill_between(df.index, threshold, ax.get_ylim()[1],
                    color='red', alpha=0.1, label='Zone critique')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Minutes depuis 00:00' if i == 3 else '')
    ax.set_ylabel(title.split('(')[1].rstrip(')'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Afficher les heures toutes les 2h
    if i == 3:
        tick_positions = range(0, 1440, 120)
        tick_labels = [f"{h:02d}:00" for h in range(0, 24, 2)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

plt.tight_layout()
plt.savefig('output/1_evolution_metriques.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: '1_evolution_metriques.png'")
plt.show()

# ============================================================================
# 4. DISTRIBUTIONS ET TESTS DE NORMALITÉ
# ============================================================================
print("\n" + "=" * 80)
print("📊 Étape 4: ANALYSE DES DISTRIBUTIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Distribution des Métriques et Tests de Normalité',
             fontsize=16, fontweight='bold')

for i, (metric, color) in enumerate(zip(metrics, colors)):
    # Histogramme
    ax1 = axes[0, i]
    ax1.hist(df[metric], bins=30, color=color, alpha=0.7, edgecolor='black')
    ax1.axvline(df[metric].mean(), color='red', linestyle='--',
                linewidth=2, label='Moyenne')
    ax1.axvline(df[metric].median(), color='orange', linestyle='--',
                linewidth=2, label='Médiane')
    ax1.set_title(metric.replace('_', ' '), fontweight='bold')
    ax1.set_xlabel('Valeur')
    ax1.set_ylabel('Fréquence')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Q-Q Plot
    ax2 = axes[1, i]
    stats.probplot(df[metric], dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Test de Shapiro-Wilk
    stat, p_value = stats.shapiro(df[metric].sample(min(5000, len(df))))
    normal = "✅ Normale" if p_value > 0.05 else "❌ Non-normale"
    ax2.text(0.05, 0.95, f'p-value: {p_value:.4f}\n{normal}',
             transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round',
                                                facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('output/2_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: '2_distributions.png'")
plt.show()

print("\n📊 Tests de Normalité (Shapiro-Wilk):")
for metric in metrics:
    stat, p_value = stats.shapiro(df[metric].sample(min(5000, len(df))))
    result = "NORMALE ✅" if p_value > 0.05 else "NON-NORMALE ❌"
    print(f"   {metric:20s}: p-value = {p_value:.4f} → {result}")

# ============================================================================
# 5. MATRICE DE CORRÉLATION
# ============================================================================
print("\n" + "=" * 80)
print("🔗 Étape 5: ANALYSE DES CORRÉLATIONS")
print("=" * 80)

correlation_matrix = df[metrics].corr()

print("\n📊 Matrice de Corrélation:\n")
print(correlation_matrix.round(3))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0.5, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            vmin=0, vmax=1, ax=ax)
ax.set_title('Matrice de Corrélation des Métriques',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('output/3_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé: '3_correlation_matrix.png'")
plt.show()

print("\n💡 Corrélations Fortes (|r| > 0.7):")
for i, metric1 in enumerate(metrics):
    for j, metric2 in enumerate(metrics):
        if i < j:
            corr = correlation_matrix.loc[metric1, metric2]
            if abs(corr) > 0.7:
                sign = "positive" if corr > 0 else "négative"
                print(f"   • {metric1} ↔ {metric2}: {corr:.3f} (corrélation {sign})")

# ============================================================================
# 6. DÉTECTION D'ANOMALIES
# ============================================================================
print("\n" + "=" * 80)
print("⚠️  Étape 6: DÉTECTION D'ANOMALIES")
print("=" * 80)

anomaly_results = {}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Détection d\'Anomalies - Z-Score et IQR',
             fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    values = df[metric]

    # Méthode Z-Score
    z_scores = np.abs(stats.zscore(values))
    z_anomalies = z_scores > 3

    # Méthode IQR
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_anomalies = (values < lower_bound) | (values > upper_bound)

    # Seuils de vigilance
    thresholds_dict = {'CPU_Usage': 80, 'Memory_Usage': 12,
                       'Network_Usage': 150, 'Temperature': 70}
    threshold = thresholds_dict[metric]
    threshold_exceeded = values > threshold

    anomaly_results[metric] = {
        'Z-Score (>3)': z_anomalies.sum(),
        'IQR': iqr_anomalies.sum(),
        f'Seuil >{threshold}': threshold_exceeded.sum(),
        'Pourcentage': (threshold_exceeded.sum() / len(df)) * 100
    }

    # Visualisation
    ax = axes[idx]
    ax.plot(df.index, values, color=color, alpha=0.6, linewidth=1)
    ax.scatter(df.index[z_anomalies], values[z_anomalies],
               color='red', s=50, label=f'Z-Score ({z_anomalies.sum()})',
               marker='o', alpha=0.8, edgecolors='darkred')
    ax.scatter(df.index[iqr_anomalies], values[iqr_anomalies],
               color='orange', s=30, label=f'IQR ({iqr_anomalies.sum()})',
               marker='s', alpha=0.8)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Seuil: {threshold}')
    ax.axhline(upper_bound, color='orange', linestyle=':', linewidth=1.5,
               label=f'IQR bounds', alpha=0.7)
    ax.axhline(lower_bound, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_title(metric.replace('_', ' '), fontweight='bold')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Valeur')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/4_anomalies_detection.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: '4_anomalies_detection.png'")
plt.show()

print("\n📊 Résumé des Anomalies Détectées:\n")
anomaly_df = pd.DataFrame(anomaly_results).T
print(anomaly_df.round(2))

# ============================================================================
# 7. MODÈLES DE PRÉDICTION (RÉGRESSION LINÉAIRE)
# ============================================================================
print("\n" + "=" * 80)
print("🔮 Étape 7: MODÈLES DE PRÉDICTION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Prédictions avec Régression Linéaire',
             fontsize=16, fontweight='bold')
axes = axes.flatten()

prediction_results = {}

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metric].values

    # Modèle de régression
    model = LinearRegression()
    model.fit(X, y)

    # Prédictions actuelles et futures
    X_future = np.arange(len(df) + 60).reshape(-1, 1)
    y_pred = model.predict(X_future)

    # Métriques
    r2_score = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    if abs(slope) > 0.01:
        trend = "Hausse" if slope > 0 else "Baisse"
    else:
        trend = "Stable"

    prediction_results[metric] = {
        'Pente': slope,
        'Ordonnée': intercept,
        'R²': r2_score,
        'Tendance': trend
    }

    # Visualisation
    ax = axes[idx]
    ax.scatter(X, y, alpha=0.3, s=10, color=color, label='Données réelles')
    ax.plot(X, model.predict(X), color='red', linewidth=2,
            label='Régression linéaire')
    ax.plot(X_future[-60:], y_pred[-60:], color='orange', linewidth=2,
            linestyle='--', label='Prédiction 60 min')
    ax.axvline(len(df), color='green', linestyle=':', linewidth=2,
               label='Limite actuelle')

    ax.set_title(f"{metric.replace('_', ' ')} - {trend}", fontweight='bold')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Valeur')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotation
    textstr = f'y = {slope:.4f}x + {intercept:.2f}\nR² = {r2_score:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round',
                                               facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('output/5_predictions_regression.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: '5_predictions_regression.png'")
plt.show()

print("\n📊 Résultats des Modèles de Prédiction:\n")
pred_df = pd.DataFrame(prediction_results).T
print(pred_df.round(4))

# ============================================================================
# 8. ANALYSE DES PATTERNS TEMPORELS
# ============================================================================
print("\n" + "=" * 80)
print("⏰ Étape 8: ANALYSE DES PATTERNS TEMPORELS")
print("=" * 80)


df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['Hour'] = df['Time'].dt.hour
hourly_stats = df.groupby('Hour')[metrics].agg(['mean', 'std', 'max'])

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Patterns Temporels - Analyse par Heure',
             fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[idx]
    hours = range(24)
    means = hourly_stats[metric]['mean']
    stds = hourly_stats[metric]['std']
    maxs = hourly_stats[metric]['max']

    ax.plot(hours, means, color=color, linewidth=3, marker='o',
            markersize=8, label='Moyenne')
    ax.fill_between(hours, means - stds, means + stds,
                    color=color, alpha=0.2, label='±1 écart-type')
    ax.plot(hours, maxs, color='red', linewidth=2, linestyle='--',
            marker='s', markersize=6, label='Maximum', alpha=0.7)

    # Identifier les heures de pic
    peak_hour = means.idxmax()
    peak_value = means.max()
    ax.axvline(peak_hour, color='red', linestyle=':', alpha=0.5)
    ax.text(peak_hour, peak_value, f'Pic: {peak_hour}h',
            fontsize=9, ha='center', bbox=dict(boxstyle='round',
                                               facecolor='yellow', alpha=0.5))

    ax.set_title(metric.replace('_', ' '), fontweight='bold')
    ax.set_xlabel('Heure de la journée')
    ax.set_ylabel('Valeur')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/6_temporal_patterns.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: '6_temporal_patterns.png'")
plt.show()

print("\n📊 Heures de Pic Identifiées:")
for metric in metrics:
    peak_hour = hourly_stats[metric]['mean'].idxmax()
    peak_value = hourly_stats[metric]['mean'].max()
    print(f"   • {metric:20s}: {peak_hour:02d}:00 ({peak_value:.2f})")

# ============================================================================
# 9. EXPORT DES RÉSULTATS
# ============================================================================
print("\n📁 Étape 10: Export des résultats...")

# Sauvegarder les statistiques
stats_results.T.to_csv('output/resultats_statistiques.csv')
print("✅ Statistiques exportées: 'resultats_statistiques.csv'")

# Sauvegarder les anomalies
anomaly_df.to_csv('output/resultats_anomalies.csv')
print("✅ Anomalies exportées: 'resultats_anomalies.csv'")

# Sauvegarder la matrice de corrélation
correlation_matrix.to_csv('output/matrice_correlation.csv')
print("✅ Corrélations exportées: 'matrice_correlation.csv'")

print("\n" + "=" * 80)
print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
print("=" * 80)
print("📊 6 graphiques générés")
print("📁 3 fichiers CSV exportés")
print("=" * 80)