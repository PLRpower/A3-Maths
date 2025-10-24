"""
Déchiffreur de César pour Messages Numériques
Château MacLeod - Étape 1
"""

def dechiffrer_cesar(message_chiffre, decalage):
    """
    Déchiffre un message numérique avec le chiffre de César.

    Args:
        message_chiffre (str): Le message chiffré (chiffres uniquement)
        decalage (int): Le décalage utilisé pour le chiffrement

    Returns:
        str: Le message déchiffré
    """
    message_dechiffre = ""

    for caractere in message_chiffre:
        if caractere.isdigit():
            # Convertir en nombre
            chiffre = int(caractere)
            # Appliquer le décalage inverse (modulo 10 pour rester dans 0-9)
            chiffre_dechiffre = (chiffre - decalage) % 10
            message_dechiffre += str(chiffre_dechiffre)
        else:
            # Conserver les caractères non-numériques (espaces, ponctuation)
            message_dechiffre += caractere

    return message_dechiffre


def formater_coordonnees_gps(message_dechiffre):
    # Vérifier la longueur minimale requise
    # Format: XXYYZZZZA BCCDDDD(E)
    # Latitude: XX°YY'ZZ.Z"A (positions 0-8)
    # Longitude: B°CC'DD.D"E (positions 9-16)

    modele = 'XX°XX\'XX.X"N X°XX\'XX.X"W'

    resultat = modele
    for c in message_dechiffre:
        resultat = resultat.replace('X', c, 1)  # remplace le premier X trouvé

    return resultat


def analyser_cesar_brute_force(message_chiffre):
    """
    Teste tous les décalages possibles (0-9) pour aider à trouver le bon.
    """
    print("=== Analyse par Force Brute ===\n")
    for decalage in range(10):
        resultat = dechiffrer_cesar(message_chiffre, decalage)
        print(f"Décalage {decalage}: {resultat}")
    print()


# ========== DÉMONSTRATION SCIENTIFIQUE ==========

print("=" * 60)
print("CHÂTEAU MACLEOD - DÉCHIFFREMENT DES COORDONNÉES GPS")
print("=" * 60)
print()

# Message du parchemin
message_chiffre = "9153787770964"

print("📜 MESSAGE CHIFFRÉ:")
print(f"   {message_chiffre}")
print()

decalage = 4
message_dechiffre = dechiffrer_cesar(message_chiffre, decalage)

print("✅ MESSAGE DÉCHIFFRÉ:")
print(f"   {message_dechiffre}")
print()

print("🗺️  COORDONNÉES GPS:")
coordonnees = formater_coordonnees_gps(message_dechiffre)
print(f"   {coordonnees}")
print()