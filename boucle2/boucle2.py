import pandas as pd
import matplotlib.pyplot as plt

# Préparer les données
df = pd.read_csv('day.csv')

# Données
#      instant      dteday  season  yr  mnth  holiday  weekday  workingday  \
# 0          1  2011-01-01       1   0     1        0        6           0
# 1          2  2011-01-02       1   0     1        0        0           0
# ..       ...         ...     ...  ..   ...      ...      ...         ...
# 729      730  2012-12-30       1   1    12        0        0           0
# 730      731  2012-12-31       1   1    12        0        1           1

#      weathersit      temp     atemp       hum  windspeed  casual  registered    cnt
# 0             2  0.344167  0.363625  0.805833   0.160446     331         654    985
# 1             2  0.363478  0.353739  0.696087   0.248539     131         670    801
# ..          ...       ...       ...       ...        ...     ...         ...    ...
# 729           1  0.255833  0.231700  0.483333   0.350754     364        1432    1796
# 730           2  0.215833  0.223487  0.577500   0.154846     439        2290    2729

# Par jour de la semaine
plt.figure(figsize=(10, 6))
plt.boxplot(
    [df[df['weekday'] == i]['cnt'] for i in sorted(df['weekday'].unique())],
    labels=['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
)
plt.title("Répartition du nombre de locations de vélos par jour de la semaine")
plt.xlabel("Jour de la semaine")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Par saison
plt.figure(figsize=(8, 5))
plt.boxplot(
    [df[df['season'] == i]['cnt'] for i in sorted(df['season'].unique())],
    labels=['Printemps', 'Été', 'Automne', 'Hiver']
)
plt.title("Répartition des locations par saison")
plt.xlabel("Saison")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Par météo
plt.figure(figsize=(8, 5))
plt.boxplot(
    [df[df['weathersit'] == i]['cnt'] for i in sorted(df['weathersit'].unique())],
    labels=['Beau', 'Nuageux', 'Pluvieux']
)
plt.title("Répartition des locations selon la météo")
plt.xlabel("Situation météo")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Par mois
plt.figure(figsize=(12, 6))
plt.boxplot(
    [df[df['mnth'] == i]['cnt'] for i in sorted(df['mnth'].unique())],
    labels=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
)
plt.title("Répartition des locations par mois")
plt.xlabel("Mois")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Par année
plt.figure(figsize=(8, 5))
plt.boxplot(
    [df[df['yr'] == i]['cnt'] for i in sorted(df['yr'].unique())],
    labels=['2011', '2012']
)
plt.title("Répartition des locations par année")
plt.xlabel("Année")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Si c'est un jour de travail ou pas
plt.figure(figsize=(8, 5))
plt.boxplot(
    [df[df['workingday'] == 0]['cnt'], df[df['workingday'] == 1]['cnt']],
    labels=['Non jour de travail', 'Jour de travail']
)
plt.title("Répartition des locations selon les jours de travail")
plt.xlabel("Type de jour")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Par rapport à la température ressentie (atemp)
cols_meteo = ['atemp', 'hum', 'windspeed']
df['atemp'] = df['atemp'] * 10000

plt.figure(figsize=(12, 6))
plt.plot(df['dteday'], df['cnt'], color='b', label='Nombre total de locations (cnt)')
plt.scatter(df['dteday'], df['atemp'], color='r', label='Température ressentie (atemp)')
xticks = df['dteday'][::30]
plt.xticks(xticks, rotation=45)
plt.title('Nombre total de locations de vélos par jour')
plt.xlabel('Date')
plt.ylabel('Nombre total de locations (cnt)')
plt.legend()
plt.tight_layout()
plt.show()

# Par température
df['temp'] = df['temp'] * 41

plt.figure(figsize=(10, 6))
plt.scatter(df['temp'], df['cnt'], alpha=0.5, color='g')
plt.title("Nombre de locations en fonction de la température")
plt.xlabel("Température (temp)")
plt.ylabel("Nombre de locations (cnt)")
plt.grid(linestyle='--', alpha=0.7)
plt.show()
