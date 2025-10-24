import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ------- Lecture et mise en forme du fichier CSV -------

csv_files = sorted(glob.glob("donnees/task_intervals_1.csv"))
dfs = [pd.read_csv(f) for f in csv_files]
tasks_df = pd.concat(dfs, ignore_index=True)

tasks_df['Start'] = tasks_df['Start'].astype(float)
tasks_df['End'] = tasks_df['End'].astype(float)


# ------- Construction du graphe d'intervalles -------

# Création d’un graphe non orienté : chaque tâche sera un sommet
G = nx.Graph()

# Ajout des nœuds représentant les tâches, avec les intervalles en attributs
for idx, row in tasks_df.iterrows():
    G.add_node(row['Task'], start=row['Start'], end=row['End'])

# Création des arêtes : deux tâches sont reliées si leurs intervalles se chevauchent
# Autrement dit : si elles ne peuvent pas être exécutées sur le même serveur.
tasks = tasks_df.to_dict('records')
for i in range(len(tasks)):
    for j in range(i + 1, len(tasks)):
        # Condition de chevauchement : les intervalles se croisent
        # On ajoute une arête si A commence avant la fin de B ET B commence avant la fin de A
        if not (tasks[i]['End'] <= tasks[j]['Start'] or tasks[j]['End'] <= tasks[i]['Start']):
            G.add_edge(tasks[i]['Task'], tasks[j]['Task'])


# ------- Coloration gloutonne  -------

# Objectif : attribuer une "couleur" (serveur) à chaque sommet (tâche)

# 1. Trier les sommets selon une certaine stratégie (ici 'largest_first' = degré décroissant)
# 2. Pour chaque sommet dans cet ordre :
#       - examiner les couleurs des voisins déjà colorés
#       - attribuer la plus petite couleur disponible (le plus petit entier non utilisé par ses voisins)
# C’est une approche heuristique (pas toujours optimale),
# mais elle produit une bonne approximation, surtout pour les graphes d’intervalles.

color_map = nx.coloring.greedy_color(G, strategy='largest_first')

unique_colors = sorted(set(color_map.values()))
num_servers = len(unique_colors)
color_list = list(plt.cm.tab20.colors)
node_colors = [color_list[color_map[node]] for node in G.nodes()]


# ------- 1. Visualisation du graphe de chevauchement -------

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_weight='bold')

plt.title(f"Graphe d'intervalles (tâches et chevauchements)\nNombre de serveurs : {num_servers}")
plt.show()

# ------- 2. Visualisation temporelle des intervalles colorés -------

plt.figure(figsize=(12, 6))

for idx, row in tasks_df.iterrows():
    plt.barh(row['Task'], row['End'] - row['Start'], left=row['Start'], color=color_list[color_map[row['Task']]], edgecolor='black')

plt.xlabel("Temps")
plt.ylabel("Tâches")
plt.title(f"Diagramme d'intervalles des tâches (serveurs colorés)\nNombre de serveurs : {num_servers}")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()

# ------- 3. Affichage textuel de la répartition des tâches par serveur -------

print("\nRépartition des tâches par serveur :")

for server_id in unique_colors:
    tasks_on_server = [task for task, color in color_map.items() if color == server_id]
    print(f"Serveur {server_id + 1} : {len(tasks_on_server)} tâches -> {tasks_on_server}")
