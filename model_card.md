# Model Card — Sentiment Classification

## 1. Use Case
Classification binaire des sentiments d'avis clients
en francais.

Classe 1 : sentiment positif.
Classe 0 : sentiment negatif.

## 2. Version
Nom du modele : sentiment-model
Version MLflow : 1
Experiment : sentiment-classification

## 3. Dataset
Fichier : data/reviews.csv

20 avis clients pedagogiques.
10 avis positifs et 10 avis negatifs.

Repartition :
- Entrainement : 75 %
- Test : 25 %
- Random state : 42

## 4. Method
TF-IDF + Logistic Regression.

Ameliorations :
- Normalisation des accents
- Suppression de mots neutres
- Equilibrage des classes

## 5. Parameters
C = 1.0
max_features = 5000
ngram_range = [1, 2]
max_iter = 1000

## 6. Metrics
Accuracy = 1.0 (100 %)
F1-score = 1.0 (100 %)

## 7. Quality Gates
Accuracy minimale : 0.60
F1 minimal : 0.50
Couverture minimale : 70 %

Tests Pytest : 9/9 PASSED
Couverture : 100 %
Ruff : All checks passed
Bandit : 0 alerte sur 212 lignes

Non-regression : regle a completer.

## 8. Security
Analyse statique realisee avec Bandit.

Un audit local OSV a signale des vulnerabilites
dans certaines dependances. La configuration CI
utilise des contraintes de versions correctives.

Le workflow GitHub Actions a termine avec succes.

## 9. Governance
validation_status = passed
approval_status = approved
candidate = version 1
champion = version 1

La promotion a ete simulee avec :
python promote.py --approve

## 10. Known Limitations
- Dataset de seulement 20 avis.
- Evaluation sur seulement 5 avis de test.
- Resultats non representatifs du contexte industriel.
- Langue principalement francaise.
- Approbation humaine simulee.
- Deploiement staging et production simule.
- Validation de non-regression non encore automatisee.

## 11. Deployment Decision
La version 1 a satisfait les tests et les seuils
de performance pedagogiques.

Le modele a ete enregistre dans MLflow, puis
promu de candidate vers champion apres approbation
simulee.

Aucun service de prediction n'a ete deploye
en production reelle.
