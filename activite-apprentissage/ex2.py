# =============================================================================
# Activité d'apprentissage - Exercice 2
# Calcul de moyenne avec nombre de notes variable
# =============================================================================

def exercice2():
    print("\n" + "=" * 60)
    print("EXERCICE 2 - Calcul de moyenne des notes")
    print("=" * 60 + "\n")

    print("Saisissez les notes une par une.")
    print("Entrez -1 pour terminer la saisie.\n")

    # Variables nécessaires
    somme_notes = 0  # Accumulation des notes
    nombre_notes = 0  # Compteur de notes

    # Structure TANT QUE : on ne connaît pas le nombre d'itérations à l'avance
    # La condition de sortie dépend de la valeur saisie
    while True:
        try:
            note = float(input(f"Note {nombre_notes + 1} : "))

            # Condition de sortie
            if note == -1:
                break

            # Validation de la note
            if note < 0 or note > 20:
                print("La note doit être entre 0 et 20. Réessayez.")
                continue

            # Accumulation
            somme_notes += note
            nombre_notes += 1

        except ValueError:
            print("Veuillez entrer un nombre valide.")

    # Calcul et affichage de la moyenne
    print("\n" + "-" * 60)
    if nombre_notes > 0:
        moyenne = somme_notes / nombre_notes
        print(f"Nombre de notes saisies : {nombre_notes}")
        print(f"Moyenne : {moyenne:.2f}/20")

        # Appréciation
        if moyenne >= 16:
            print("Mention : Très bien")
        elif moyenne >= 14:
            print("Mention : Bien")
        elif moyenne >= 12:
            print("Mention : Assez bien")
        elif moyenne >= 10:
            print("Mention : Passable")
        else:
            print("Mention : Insuffisant")
    else:
        print("Aucune note n'a été saisie.")

    print("\nJustification du choix de structure :")
    print("→ TANT QUE est idéal car :")
    print("  1. Le nombre de notes n'est pas connu à l'avance")
    print("  2. La sortie dépend d'une condition (saisie de -1)")
    print("  3. On peut gérer le cas où aucune note n'est saisie")

