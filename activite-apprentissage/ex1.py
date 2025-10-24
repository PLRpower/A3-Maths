# =============================================================================
# Activité d'apprentissage - Exercice 1
# Programme de bienvenue avec vérification d'âge
# =============================================================================


def exercice1():
    print("\n" + "=" * 60)
    print("Exercice 1")
    print("=" * 60 + "\n")

    # Saisie du nom
    nom = input("Entrez votre nom : ")

    # Saisie de l'âge avec validation
    while True:
        try:
            age = int(input("Entrez votre âge : "))
            if age < 0:
                print("L'âge ne peut pas être négatif. Réessayez.")
                continue
            break
        except ValueError:
            print("Veuillez entrer un nombre valide.")

    # Message de bienvenue
    print(f"\nBienvenue {nom} !")

    # Vérification de la majorité
    if age >= 18:
        print("Vous êtes majeur(e)")
    else:
        print("Vous êtes mineur(e)")


exercice1()
