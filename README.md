# Analyse Différentielle sur un Schéma de Feistel

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Ekole237/Freistel)](https://github.com/Ekole237/Freistel/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/Ekole237/Freistel)](https://github.com/Ekole237/Freistel/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/Ekole237/Freistel)](https://github.com/Ekole237/Freistel/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/Ekole237/Freistel)](https://github.com/Ekole237/Freistel/commits/main)

Ce projet implémente une simulation d'attaque par analyse différentielle sur un schéma de Feistel, permettant d'étudier les vulnérabilités potentielles des algorithmes de chiffrement basés sur cette structure.

## 🎯 Objectifs du Projet

- Simuler une attaque différentielle sur un algorithme de chiffrement basé sur un schéma de Feistel
- Analyser les résultats pour identifier les faiblesses potentielles
- Générer des rapports détaillés avec visualisations et analyses approfondies
- Fournir des explications accessibles pour les experts et non-experts

## 🔒 Concepts Cryptographiques

### Schéma de Feistel
Le schéma de Feistel est une structure utilisée dans de nombreux algorithmes de chiffrement par bloc (comme DES). Il divise le bloc d'entrée en deux moitiés et applique une série de transformations répétées appelées "tours". Chaque tour utilise une fonction F et une sous-clé pour transformer les données.

### Analyse Différentielle
L'analyse différentielle est une technique de cryptanalyse qui :
- Étudie comment les différences dans les entrées affectent les différences en sortie
- Exploite les biais statistiques dans la distribution des différences
- Peut révéler des faiblesses dans la conception du chiffrement

## 🛠️ Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/Ekole237/Freistel.git
cd Freistel
```

2. Créer et activer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer l'API OpenAI :
   - Copier le fichier `.env.example` vers `.env`
   - Ajouter votre clé API OpenAI dans le fichier `.env`

## 🚀 Utilisation

### Exécution de l'Analyse

```bash
python src/main.py
```

Cette commande :
1. Initialise un chiffrement de Feistel avec des paramètres configurables
2. Génère des paires de textes avec des différences spécifiques
3. Analyse la propagation des différences à travers le chiffrement
4. Collecte des statistiques sur les différences en sortie

### Génération du Rapport

```bash
python src/generate_report.py
```

Le rapport généré inclut :
- Une introduction théorique aux schémas de Feistel
- Les détails de l'attaque différentielle réalisée
- Des visualisations interactives des résultats
- Une analyse approfondie générée par GPT-4
- Des explications adaptées aux différents niveaux d'expertise

## 📊 Structure du Projet

```
Freistel/
├── src/
│   ├── feistel.py          # Implémentation du schéma de Feistel
│   ├── differential.py     # Analyse différentielle
│   ├── report_generator.py # Générateur de rapports
│   ├── main.py            # Point d'entrée principal
│   └── generate_report.py  # Script de génération
├── results/               # Rapports et visualisations
├── requirements.txt       # Dépendances
└── .env                  # Configuration (API keys)
```

### Composants Principaux

#### 1. Schéma de Feistel (`feistel.py`)
- Implémentation modulaire et configurable
- Support de différentes fonctions de tour
- Paramètres ajustables (nombre de tours, taille de bloc)

#### 2. Analyse Différentielle (`differential.py`)
- Génération de paires de textes avec différences contrôlées
- Analyse statistique des différences en sortie
- Détection des biais cryptographiques

#### 3. Génération de Rapports (`report_generator.py`)
- Visualisations détaillées des résultats
- Intégration avec GPT-4 pour les analyses
- Support multilingue (français/anglais)

## 📈 Visualisations

Le projet génère deux types de visualisations principales :

### 1. Distribution des Différences
- Histogramme des différences en sortie
- Mise en évidence des biais statistiques
- Annotations explicatives des pics significatifs

### 2. Analyse des Probabilités
- Comparaison avec la distribution théorique
- Échelle logarithmique pour une meilleure lisibilité
- Calcul et affichage du facteur de biais

## 🔍 Résultats Typiques

L'analyse permet de :
- Identifier les différences qui se propagent avec des probabilités non uniformes
- Quantifier la résistance du chiffrement aux attaques différentielles
- Suggérer des améliorations pour renforcer la sécurité

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forker le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

### Domaines d'Amélioration
- Ajout de nouveaux types d'analyses
- Optimisation des performances
- Amélioration des visualisations
- Documentation multilingue

## 📚 Références

1. Biham, E., & Shamir, A. (1991). Differential cryptanalysis of DES-like cryptosystems.
2. Feistel, H. (1973). Cryptography and Computer Privacy.
3. [Documentation OpenAI](https://platform.openai.com/docs/api-reference)
4. [Documentation Matplotlib](https://matplotlib.org/)

## 📜 Licence

Ce projet est sous licence [MIT](LICENSE).

---

*Développé dans le cadre d'un projet de recherche en cryptanalyse*
