# =============================================================================
# Activité d'apprentissage - Exercice 3
# Système de gestion d'inventaire
# =============================================================================

class Produit:
    """Classe représentant un produit dans l'inventaire"""

    def __init__(self, nom, quantite, prix):
        self.nom = nom
        self.quantite = quantite
        self.prix = prix

    def __str__(self):
        valeur = self.quantite * self.prix
        return f"{self.nom:20s} | Qté: {self.quantite:5d} | Prix: {self.prix:7.2f}€ | Valeur: {valeur:8.2f}€"


def afficher_menu():
    """Affiche le menu principal"""
    print("\n" + "=" * 60)
    print("SYSTÈME DE GESTION D'INVENTAIRE")
    print("=" * 60)
    print("1. Ajouter un produit")
    print("2. Rechercher un produit")
    print("3. Modifier la quantité d'un produit")
    print("4. Calculer la valeur totale de l'inventaire")
    print("5. Afficher tous les produits")
    print("6. Quitter")
    print("=" * 60)


def ajouter_produit(inventaire):
    """Ajoute un nouveau produit à l'inventaire"""
    print("\n--- Ajouter un produit ---")

    nom = input("Nom du produit : ").strip()
    if not nom:
        print("❌ Le nom ne peut pas être vide.")
        return

    # Vérifier si le produit existe déjà
    for produit in inventaire:
        if produit.nom.lower() == nom.lower():
            print(f"❌ Le produit '{nom}' existe déjà dans l'inventaire.")
            return

    try:
        quantite = int(input("Quantité : "))
        if quantite < 0:
            print("❌ La quantité ne peut pas être négative.")
            return

        prix = float(input("Prix unitaire (€) : "))
        if prix < 0:
            print("❌ Le prix ne peut pas être négatif.")
            return

        inventaire.append(Produit(nom, quantite, prix))
        print(f"✓ Produit '{nom}' ajouté avec succès !")

    except ValueError:
        print("❌ Erreur : veuillez entrer des valeurs numériques valides.")


def rechercher_produit(inventaire):
    """Recherche un produit par son nom"""
    print("\n--- Rechercher un produit ---")

    nom = input("Nom du produit à rechercher : ").strip()

    for produit in inventaire:
        if produit.nom.lower() == nom.lower():
            print("\n✓ Produit trouvé :")
            print("-" * 60)
            print(produit)
            return produit

    print(f"❌ Produit '{nom}' non trouvé dans l'inventaire.")
    return None


def modifier_quantite(inventaire):
    """Modifie la quantité d'un produit"""
    print("\n--- Modifier la quantité ---")

    nom = input("Nom du produit : ").strip()

    for produit in inventaire:
        if produit.nom.lower() == nom.lower():
            print(f"Quantité actuelle : {produit.quantite}")

            try:
                nouvelle_quantite = int(input("Nouvelle quantité : "))
                if nouvelle_quantite < 0:
                    print("❌ La quantité ne peut pas être négative.")
                    return

                ancienne_quantite = produit.quantite
                produit.quantite = nouvelle_quantite
                print(f"✓ Quantité modifiée : {ancienne_quantite} → {nouvelle_quantite}")

            except ValueError:
                print("❌ Veuillez entrer un nombre valide.")
            return

    print(f"❌ Produit '{nom}' non trouvé.")


def calculer_valeur_totale(inventaire):
    """Calcule la valeur totale de l'inventaire"""
    print("\n--- Valeur totale de l'inventaire ---")

    if not inventaire:
        print("L'inventaire est vide.")
        return

    valeur_totale = sum(p.quantite * p.prix for p in inventaire)
    nombre_produits = len(inventaire)
    quantite_totale = sum(p.quantite for p in inventaire)

    print(f"Nombre de produits différents : {nombre_produits}")
    print(f"Quantité totale d'articles : {quantite_totale}")
    print(f"Valeur totale : {valeur_totale:.2f}€")


def afficher_inventaire(inventaire):
    """Affiche tous les produits de l'inventaire"""
    print("\n--- Inventaire complet ---")

    if not inventaire:
        print("L'inventaire est vide.")
        return

    print("-" * 60)
    print(f"{'PRODUIT':20s} | {'QUANTITÉ':^11s} | {'PRIX':^11s} | {'VALEUR':^11s}")
    print("-" * 60)

    for produit in inventaire:
        print(produit)

    print("-" * 60)
    print(f"Total de produits : {len(inventaire)}")


def exercice3():
    """Programme principal de gestion d'inventaire"""
    print("\n" + "=" * 60)
    print("EXERCICE 3 - Système de gestion d'inventaire")
    print("=" * 60 + "\n")

    inventaire = []

    # Boucle principale du programme
    while True:
        afficher_menu()

        try:
            choix = input("\nVotre choix (1-6) : ").strip()

            if choix == "1":
                ajouter_produit(inventaire)

            elif choix == "2":
                rechercher_produit(inventaire)

            elif choix == "3":
                modifier_quantite(inventaire)

            elif choix == "4":
                calculer_valeur_totale(inventaire)

            elif choix == "5":
                afficher_inventaire(inventaire)

            elif choix == "6":
                print("\n✓ Au revoir !")
                break

            else:
                print("❌ Choix invalide. Veuillez entrer un nombre entre 1 et 6.")

        except KeyboardInterrupt:
            print("\n\n✓ Programme interrompu. Au revoir !")
            break


exercice3()
