# ------------------------------------------------------------
# Modélisation d’un jeu entre deux joueurs : Marc et Nicole
# ------------------------------------------------------------
# Objectif :
# - Marc cherche à minimiser son coût (valeur la plus faible)
# - Nicole cherche à maximiser son profit (valeur la plus élevée)
# - Le paramètre p représente une probabilité (ex. événement aléatoire)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Fonction des gains espérés pour chaque combinaison de choix
# ------------------------------------------------------------
def gains_esperes(p):
    """
    Calcule les gains (ou coûts) espérés pour chaque couple de stratégies (Marc, Nicole).

    Retour :
        Un dictionnaire où chaque clé représente une combinaison d’actions :
            - M50_N50 : Marc choisit 50, Nicole choisit 50
            - M50_N70 : Marc choisit 50, Nicole choisit 70
            - M70_N50 : Marc choisit 70, Nicole choisit 50
            - M70_N70 : Marc choisit 70, Nicole choisit 70
        Chaque valeur est un tuple (gain_Marc, gain_Nicole)
    """
    return {
        'M50_N50': (50, 50 - 55*p),
        'M50_N70': (150*p, 0),
        'M70_N50': (50, -5),
        'M70_N70': (70, 15),
    }


# ------------------------------------------------------------
# 2. Détermination des meilleures réponses de Marc et de Nicole
# ------------------------------------------------------------
def meilleures_choix(p):
    """
    Détermine, pour un certain p, les meilleures réponses de Marc et Nicole
    selon les gains espérés.

    Pour Marc :
        -> Il choisit la stratégie qui minimise son coût.
    Pour Nicole :
        -> Elle choisit la stratégie qui maximise son profit.
    """
    gains = gains_esperes(p)

    # --- Sous-fonction : meilleure réponse de Marc ---
    def meilleur_choix_marc(nicole):
        """
        Retourne la ou les meilleures réponses de Marc
        en fonction du choix de Nicole (50 ou 70).
        """
        # Coûts de Marc selon son choix (50 ou 70)
        options = {
            50: gains[f"M50_N{nicole}"][0],
            70: gains[f"M70_N{nicole}"][0],
        }

        # Marc cherche à minimiser son coût
        min_cout = min(options.values())

        # Retourne toutes les stratégies qui atteignent ce minimum
        return [m for m, cout in options.items() if cout == min_cout]

    # --- Sous-fonction : meilleure réponse de Nicole ---
    def meilleur_choix_nicole(marc):
        """
        Retourne la ou les meilleures réponses de Nicole
        en fonction du choix de Marc (50 ou 70).
        """
        # Profits de Nicole selon son choix (50 ou 70)
        options = {
            50: gains[f"M{marc}_N50"][1],
            70: gains[f"M{marc}_N70"][1],
        }

        # Nicole cherche à maximiser son profit
        max_profit = max(options.values())

        # Retourne toutes les stratégies qui atteignent ce maximum
        return [n for n, profit in options.items() if profit == max_profit]

    # On retourne les meilleures réponses de chaque joueur dans un dictionnaire clair
    return {
        "marc_mc_si_N50": meilleur_choix_marc(50),
        "marc_mc_si_N70": meilleur_choix_marc(70),
        "nicole_mc_si_M50": meilleur_choix_nicole(50),
        "nicole_mc_si_M70": meilleur_choix_nicole(70),
    }


# ------------------------------------------------------------
# 3. Recherche des équilibres de Nash purs
# ------------------------------------------------------------
def pure_nash(p):
    """
    Cherche les équilibres de Nash en stratégies pures pour un paramètre p donné.
    Un équilibre de Nash (m*, n*) est un couple de stratégies tel que :
        - Marc ne veut pas changer de stratégie (meilleure réponse à n*)
        - Nicole ne veut pas changer de stratégie (meilleure réponse à m*)
    """
    br = meilleures_choix(p)  # meilleures réponses pour p donné
    equilibre = []            # liste des équilibres trouvés

    # Tous les couples de stratégies possibles (Marc, Nicole)
    profils = [(50, 50), (50, 70), (70, 50), (70, 70)]

    # On vérifie pour chaque profil s'il est stable (aucun joueur n’a intérêt à dévier)
    for m, n in profils:
        # Marc à un meilleur choix à Nicole ?
        marc_ok = (m in (br['marc_mc_si_N50'] if n == 50 else br['marc_mc_si_N70']))

        # Nicole à un meilleur choix à Marc ?
        nicole_ok = (n in (br['nicole_mc_si_M50'] if m == 50 else br['nicole_mc_si_M70']))

        # Si les deux conditions sont vraies, on a un équilibre de Nash
        if marc_ok and nicole_ok:
            equilibre.append((m, n))

    return equilibre


# ------------------------------------------------------------
# 4. Exécution : affichage des équilibres selon p
# ------------------------------------------------------------
# On fait varier p de 0 à 1 (par pas de 0.01)
# Pour chaque valeur, on affiche les équilibres purs trouvés
for p in [i/100 for i in range(101)]:
    print(f"p = {p:.2f} -> équilibres purs : {pure_nash(p)}")
