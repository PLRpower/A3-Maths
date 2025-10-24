import math
from typing import List


def factoriser(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i, n // i
    return None, None


def pgcd_etendu(a, b):
    if a == 0:
        return b, 0, 1
    pgcd, x1, y1 = pgcd_etendu(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return pgcd, x, y


def inverse_modulaire(e, phi_n):
    pgcd, x, _ = pgcd_etendu(e, phi_n)
    if pgcd != 1:
        raise ValueError("L'inverse modulaire n'existe pas")
    return x % phi_n


def puissance_modulaire(base, exposant, module):
    resultat = 1
    base = base % module

    while exposant > 0:
        if exposant % 2 == 1:
            resultat = (resultat * base) % module
        exposant = exposant >> 1
        base = (base * base) % module

    return resultat


def dechiffrer_rsa(message_chiffre: List[int], n: int, e: int) -> str:
    """
    Déchiffre un message RSA complet.

    Args:
        message_chiffre: Liste des valeurs chiffrées
        n: Le module RSA
        e: L'exposant public

    Returns:
        Le message déchiffré en texte ASCII
    """
    # Étape 1: Factoriser N
    print("🔍 ÉTAPE 1: Factorisation de N")
    p, q = factoriser(n)
    print(f"   N = {n} = {p} × {q}")
    print()

    # Étape 2: Calculer φ(N)
    phi_n = (p - 1) * (q - 1)
    print("📐 ÉTAPE 2: Calcul de φ(N)")
    print(f"   φ(N) = (p-1) × (q-1) = ({p}-1) × ({q}-1)")
    print(f"   φ(N) = {p - 1} × {q - 1} = {phi_n}")
    print()

    # Étape 3: Calculer D (clé privée)
    print("🔑 ÉTAPE 3: Calcul de D (exposant privé)")
    d = inverse_modulaire(e, phi_n)
    print(f"   Trouver D tel que (D × E) mod φ(N) = 1")
    print(f"   D = {d}")
    print(f"   Vérification: ({d} × {e}) mod {phi_n} = {(d * e) % phi_n}")
    print()

    # Étape 4: Déchiffrer chaque valeur
    print("🔓 ÉTAPE 4: Déchiffrement")
    print(f"   Formule: M = C^D mod N")
    print()

    message_dechiffre = ""

    print("   Premiers exemples de déchiffrement:")
    for i, c in enumerate(message_chiffre[:3]):
        m = puissance_modulaire(c, d, n)
        caractere = chr(m)
        print(f"   C[{i}] = {c:4d} → M = {c}^{d} mod {n} = {m:3d} → '{caractere}'")
        message_dechiffre += caractere

    print(f"   ... (déchiffrement de {len(message_chiffre) - 3} autres valeurs)")
    print()

    # Déchiffrer le reste
    for c in message_chiffre[3:]:
        m = puissance_modulaire(c, d, n)
        message_dechiffre += chr(m)

    return message_dechiffre


def chiffrer_rsa(message: str, n: int, e: int) -> List[int]:
    """
    Chiffre un message en RSA (pour tester).

    Args:
        message: Le texte à chiffrer
        n: Le module RSA
        e: L'exposant public

    Returns:
        Liste des valeurs chiffrées
    """
    message_chiffre = []
    for caractere in message:
        m = ord(caractere)
        c = puissance_modulaire(m, e, n)
        message_chiffre.append(c)
    return message_chiffre


# ========== DÉMONSTRATION SCIENTIFIQUE ==========

print("=" * 70)
print("CHÂTEAU MACLEOD - DÉCHIFFREMENT RSA DU PARCHEMIN")
print("=" * 70)
print()

# Données du parchemin
C = [2726, 1313, 1992, 884, 2412, 1453, 1230, 2185, 2412, 1992, 1313, 1230,
     884, 1992, 281, 1632, 281, 2170, 1453, 1992, 1230, 2185, 2160, 1230,
     1992, 745, 1632, 1992, 612, 745, 1632, 1627, 2160, 1313, 1992, 2412,
     2185, 2160, 2923, 1313]
N = 3233
E = 17

print("📜 DONNÉES DU PARCHEMIN:")
print(f"   Message chiffré C: {len(C)} valeurs")
print(f"   Premiers éléments: {C[:10]}...")
print(f"   N (module): {N}")
print(f"   E (exposant public): {E}")
print()

print("📚 PRINCIPE DU CHIFFREMENT RSA:")
print("   RSA est un algorithme de cryptographie asymétrique.")
print("   Il utilise deux clés: une publique (N, E) et une privée (N, D).")
print()
print("   Chiffrement: C = M^E mod N")
print("   Déchiffrement: M = C^D mod N")
print()
print("   où M est le code ASCII du caractère original.")
print()

print("─" * 70)
print()

# Déchiffrement
message_dechiffre = dechiffrer_rsa(C, N, E)

print("=" * 70)
print("✨ RÉSULTAT FINAL")
print("=" * 70)
print()
print("📖 MESSAGE DÉCHIFFRÉ:")
print(f"   \"{message_dechiffre}\"")
print()

print("=" * 70)
print("APPLICATION GÉNÉRIQUE - AUTRES PARCHEMINS")
print("=" * 70)
print()


# Mode interactif
def mode_interactif():
    """Interface interactive pour déchiffrer d'autres messages RSA."""

    while True:
        print("\n" + "─" * 70)
        print("MENU:")
        print("1. Déchiffrer un nouveau message RSA")
        print("2. Chiffrer un message (test)")
        print("3. Quitter")

        choix = input("\nVotre choix (1-3): ").strip()

        if choix == '1':
            print("\n📥 DÉCHIFFREMENT D'UN NOUVEAU MESSAGE")
            print("─" * 70)

            try:
                n = int(input("Entrez N (module): "))
                e = int(input("Entrez E (exposant public): "))

                print("\nEntrez les valeurs chiffrées séparées par des virgules:")
                c_input = input("C = ")
                c_list = [int(x.strip()) for x in c_input.split(',')]

                print("\n" + "=" * 70)
                message = dechiffrer_rsa(c_list, n, e)
                print("=" * 70)
                print(f"\n✅ MESSAGE DÉCHIFFRÉ: \"{message}\"")
                print()

            except ValueError as e:
                print(f"\n⚠️  Erreur: {e}")
            except Exception as e:
                print(f"\n⚠️  Erreur lors du déchiffrement: {e}")

        elif choix == '2':
            print("\n📤 CHIFFREMENT D'UN MESSAGE (TEST)")
            print("─" * 70)

            try:
                message = input("Entrez le message à chiffrer: ")
                n = int(input("Entrez N (module): "))
                e = int(input("Entrez E (exposant public): "))

                c_list = chiffrer_rsa(message, n, e)

                print(f"\n✅ MESSAGE CHIFFRÉ:")
                print(f"C = {c_list}")
                print()

            except Exception as e:
                print(f"\n⚠️  Erreur lors du chiffrement: {e}")

        elif choix == '3':
            print("\n👋 Au revoir, explorateur du Château MacLeod!")
            print("Que vos découvertes vous mènent au trésor! 🏆")
            break

        else:
            print("\n⚠️  Choix invalide. Veuillez entrer 1, 2 ou 3.")


# Lancer le mode interactif
mode_interactif()