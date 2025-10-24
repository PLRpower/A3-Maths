import random
import time
import csv
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional


# ============================================================================
# ALGORITHME 1 : FORCE BRUTE - O(n²)
# ============================================================================
def brute_force_two_sum(surplus: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Approche par force brute avec double boucle imbriquée.

    Complexité temporelle : O(n²)
    Complexité spatiale : O(1)
    """
    n = len(surplus)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if surplus[i] + surplus[j] == target:
                return i, j
    return None

# ============================================================================
# ALGORITHME 2 : TABLE DE HACHAGE - O(n)
# ============================================================================
def hash_table_two_sum(surplus: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Approche optimisée avec table de hachage (dictionnaire Python).
    Principe : Pour chaque élément, on vérifie si son complément
    (target - élément) existe déjà dans la table.

    Complexité temporelle : O(n) - accès O(1) en moyenne
    Complexité spatiale : O(n)
    """
    hash_map = {}
    for i, value in enumerate(surplus):
        complement = target - value
        if complement in hash_map:
            return hash_map[complement], i
        hash_map[value] = i
    return None


# ============================================================================
# ALGORITHME 3 : TWO POINTERS (TRI + DEUX POINTEURS) - O(n log n)
# ============================================================================
def two_pointers_two_sum(surplus: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Approche avec tri puis deux pointeurs.

    Complexité temporelle : O(n log n) - dominé par le tri
    Complexité spatiale : O(n) - stockage des indices
    """
    n = len(surplus)
    indices = list(range(n))

    # Option 1 : faire un tri "classique"
    indices.sort(key=lambda i: surplus[i])

    # Option 2 : Faire un tri dichotomique (moins efficace en Python)
    # indices = sorted(indices, key=lambda i: surplus[i])x


    left, right = 0, n - 1

    while left < right:
        current_sum = surplus[indices[left]] + surplus[indices[right]]

        if current_sum == target:
            # Retourner les indices originaux triés
            idx1, idx2 = indices[left], indices[right]
            return min(idx1, idx2), max(idx1, idx2)
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return None


# ============================================================================
# UTILITAIRES DE LECTURE DES DONNÉES
# ============================================================================
def load_csv_data(filepath: str) -> Tuple[List[int], int]:
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        target = int(next(reader)[0])
        surplus = [int(row[0]) for row in reader]

    return surplus, target


def get_all_data_files() -> List[str]:
    pattern = os.path.join("GreenIT_data", "data_list_*.csv")
    files = glob.glob(pattern)

    def get_size(filename):
        basename = os.path.basename(filename)
        size_str = basename.replace("data_list_", "").replace(".csv", "")
        return int(size_str)

    files.sort(key=get_size)
    return files


# ============================================================================
# BENCHMARK ET MESURE DE PERFORMANCES
# ============================================================================
def benchmark_algorithm(algo_func, surplus: List[int], target: int, runs: int = 3) -> Tuple[float, Optional[Tuple[int, int]]]:
    times = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = algo_func(surplus, target)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Conversion en ms

    avg_time = sum(times) / len(times)
    return avg_time, result


def run_complete_benchmark() -> dict:
    files = get_all_data_files()

    results = {
        'sizes': [],
        'brute_force': [],
        'hash_table': [],
        'two_pointers': [],
        'solutions': []
    }

    print("Analyse des performances en cours ...")
    print()

    for filepath in files:
        # Charger les données
        surplus, target = load_csv_data(filepath)
        size = len(surplus)

        results['sizes'].append(size)

        # Test Force Brute (toujours exécuté maintenant)
        time_brute, result_brute = benchmark_algorithm(brute_force_two_sum, surplus, target)
        results['brute_force'].append(time_brute)

        # Test Table de Hachage
        time_hash, result_hash = benchmark_algorithm(hash_table_two_sum, surplus, target)
        results['hash_table'].append(time_hash)

        # Test Two Pointers
        time_pointers, result_pointers = benchmark_algorithm(two_pointers_two_sum, surplus, target)
        results['two_pointers'].append(time_pointers)

        # Vérifier la cohérence des résultats
        results['solutions'].append({
            'brute': result_brute,
            'hash': result_hash,
            'pointers': result_pointers
        })

    return results


# ============================================================================
# VISUALISATION DES RÉSULTATS
# ============================================================================
def plot_performance_comparison(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LoGreenTech Solutions - Analyse de Performance des Algorithmes', fontsize=16, fontweight='bold')

    sizes = results['sizes']
    brute_times = results['brute_force']
    hash_times = results['hash_table']
    pointer_times = results['two_pointers']

    # ---- Graphique 1 : Performance globale (échelle log) ----
    ax1 = axes[0, 0]

    ax1.plot(sizes, brute_times, 'o-', label='Force Brute O(n²)', color='#ef4444', linewidth=2, markersize=6)
    ax1.plot(sizes, hash_times, 's-', label='Table de Hachage O(n)', color='#22c55e', linewidth=2, markersize=6)
    ax1.plot(sizes, pointer_times, '^-', label='Two Pointers O(n log n)', color='#3b82f6', linewidth=2, markersize=6)

    ax1.set_xlabel('Taille du tableau (n)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Temps d\'exécution (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Réelle (échelle logarithmique)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Graphique 2 : Comparaison des 3 algorithmes (échelle linéaire) ----
    ax2 = axes[0, 1]

    ax2.plot(sizes, brute_times, 'o-', label='Force Brute', color='#ef4444', linewidth=2, markersize=6)
    ax2.plot(sizes, hash_times, 's-', label='Table de Hachage', color='#22c55e', linewidth=2, markersize=6)
    ax2.plot(sizes, pointer_times, '^-', label='Two Pointers', color='#3b82f6', linewidth=2, markersize=6)

    ax2.set_xlabel('Taille du tableau (n)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Temps d\'exécution (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Comparaison des Trois Algorithmes (échelle linéaire)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- Graphique 3 : Complexité théorique ----
    ax3 = axes[1, 0]

    # Normaliser pour la visualisation
    n_theory = np.array(sizes)
    complexity_n2 = (n_theory ** 2) / 1e9  # O(n²)
    complexity_n = n_theory / 1e3          # O(n)
    complexity_nlogn = (n_theory * np.log2(n_theory)) / 1e4  # O(n log n)

    ax3.plot(n_theory, complexity_n2, '--', label='O(n²)', color='#ef4444', linewidth=2, alpha=0.7)
    ax3.plot(n_theory, complexity_n, '--', label='O(n)', color='#22c55e', linewidth=2, alpha=0.7)
    ax3.plot(n_theory, complexity_nlogn, '--', label='O(n log n)', color='#3b82f6', linewidth=2, alpha=0.7)

    ax3.set_xlabel('Taille du tableau (n)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Opérations (normalisées)', fontsize=11, fontweight='bold')
    ax3.set_title('Croissance Théorique de la Complexité', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ---- Graphique 4 : Ratio de performance (baseline = Hash Table) ----
    ax4 = axes[1, 1]

    # Calculer les ratios par rapport à la table de hachage
    ratio_brute = [tb / th if th > 0 else 0 for tb, th in zip(brute_times, hash_times)]
    ratio_pointers = [tp / th if th > 0 else 0 for tp, th in zip(pointer_times, hash_times)]

    ax4.plot(sizes, ratio_brute, 'o-', label='Force Brute / Hash Table', color='#ef4444', linewidth=2, markersize=6)
    ax4.plot(sizes, ratio_pointers, '^-', label='Two Pointers / Hash Table', color='#3b82f6', linewidth=2, markersize=6)
    ax4.axhline(y=1, color='#22c55e', linestyle='--', linewidth=2, label='Table de Hachage (baseline = 1)', alpha=0.7)

    ax4.set_xlabel('Taille du tableau (n)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Ratio de performance', fontsize=11, fontweight='bold')
    ax4.set_title('Ratio de Performance (baseline = Hash Table)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_table(results: dict):
    """
    Affiche un tableau récapitulatif des résultats.
    """
    print("\n" + "=" * 90)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
    print("=" * 90)
    print(f"{'Taille':>10} | {'Force Brute':>15} | {'Table Hachage':>15} | " f"{'Two Pointers':>15} | {'Speedup':>10}")
    print("-" * 90)

    for i, size in enumerate(results['sizes']):
        brute = results['brute_force'][i]
        hash_t = results['hash_table'][i]
        pointer = results['two_pointers'][i]

        speedup = f"{brute / hash_t:.1f}x" if brute and hash_t else "-"

        print(f"{size:>10} | {brute:>15.3f} ms | {hash_t:>15.3f} ms | " f"{pointer:>15.3f} ms | {speedup:>10}")

    print("=" * 90)
    print()

    # Ajouter une analyse du Two Pointers
    print("💡 ANALYSE DU TWO POINTERS :")
    print("-" * 90)
    print("Le Two Pointers est souvent PLUS LENT que la force brute sur petites données car :")
    print("  1. Coût du tri Python (Timsort) : overhead significatif pour n < 10000")
    print("  2. Création et manipulation de structures supplémentaires")
    print("  3. Le brute force bénéficie mieux du cache CPU (accès mémoire séquentiels)")
    print()
    print("Le Two Pointers devient intéressant dans ces cas :")
    print("  • Données DÉJÀ triées (pas besoin de trier → O(n) direct)")
    print("  • Très grandes données (n > 100000) où le tri compense")
    print("  • Contraintes mémoire strictes (pas de HashMap possible)")
    print("=" * 90)
    print()


def print_analysis():
    print("\n" + "=" * 70)
    print("ANALYSE ET RECOMMANDATIONS POUR LOGREENTECH SOLUTIONS")
    print("=" * 70)

    print("\n📊 RÉSULTATS CLÉS :\n")

    print("1. FORCE BRUTE - O(n²)")
    print("   ✗ Complexité quadratique inacceptable")
    print("   ✗ Temps d'exécution explose avec la taille")
    print("   ✗ Non viable pour un système temps réel")
    print("   → Utilisable uniquement pour n < 1000\n")

    print("2. TABLE DE HACHAGE - O(n) ⭐ RECOMMANDÉ")
    print("   ✓ Complexité linéaire optimale")
    print("   ✓ Accès en O(1) grâce au dictionnaire Python")
    print("   ✓ Performance stable et prévisible")
    print("   ✓ Parfait pour le temps réel du Smart Grid")
    print("   → Solution recommandée pour la production\n")

    print("3. TWO POINTERS - O(n log n)")
    print("   ✓ Meilleur que force brute")
    print("   ⚠ Nécessite un tri préalable")
    print("   ⚠ Plus lent que la table de hachage")
    print("   → Alternative si contraintes mémoire strictes\n")

    print("-" * 70)
    print("\n🎯 POURQUOI LA TABLE DE HACHAGE EST PRÉFÉRABLE ?\n")

    print("La table de hachage (dictionnaire Python) offre un accès en O(1)")
    print("en moyenne, ce qui transforme le problème :\n")

    print("  • Pour chaque élément, on cherche son complément (target - élément)")
    print("  • Avec force brute : O(n) recherches × O(n) éléments = O(n²)")
    print("  • Avec hash table : O(1) recherche × O(n) éléments = O(n)\n")

    print("Exemple concret avec 10 000 capteurs :")
    print("  • Force brute : ~100 000 000 opérations")
    print("  • Hash table  : ~10 000 opérations")
    print("  → Gain de performance : ~10 000x !\n")

    print("Pour un Smart Grid nécessitant des décisions en temps réel,")
    print("cette différence est CRITIQUE pour la réactivité du système.\n")

    print("=" * 70)


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def main():
    # Lancer le benchmark
    results = run_complete_benchmark()

    # Afficher le tableau récapitulatif
    print_summary_table(results)

    # Afficher l'analyse
    print_analysis()

    # Générer les graphiques
    plot_performance_comparison(results)


if __name__ == "__main__":
    main()