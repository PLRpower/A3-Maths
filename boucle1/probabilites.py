from math import comb

p = 0.6
n = 5

# Somme des probabilités de X >= 3
prob = sum(comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(3, n + 1))

print(f"Probabilité que au moins 3 jours soient peu nuageux : {prob}")