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

## 🛠️ Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-username/Freistel.git
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

1. Exécuter l'analyse différentielle :
```bash
python src/main.py
```
Cette commande effectue une simulation de l'attaque différentielle sur le schéma de Feistel.

2. Générer un rapport détaillé :
```bash
python src/generate_report.py
```
Cette commande génère un rapport complet avec :
- Une introduction aux schémas de Feistel
- Une description détaillée de l'attaque
- Des visualisations des résultats
- Une analyse approfondie par GPT-4
- Des explications accessibles aux non-experts

## 📊 Structure du Projet

```
Freistel/
├── src/
│   ├── feistel.py          # Implémentation du schéma de Feistel
│   ├── differential.py     # Code pour l'analyse différentielle
│   ├── report_generator.py # Générateur de rapports avec visualisations
│   ├── main.py            # Point d'entrée principal
│   └── generate_report.py  # Script de génération de rapports
├── tests/                  # Tests unitaires
├── docs/                   # Documentation supplémentaire
├── results/               # Rapports et visualisations générés
├── requirements.txt       # Dépendances du projet
├── .env.example          # Exemple de configuration
└── README.md             # Ce fichier
```

## 📝 Fonctionnalités

### Schéma de Feistel
- Implémentation modulaire et configurable
- Support de différentes fonctions de tour
- Paramètres ajustables (nombre de tours, taille de bloc)

### Analyse Différentielle
- Génération automatique de paires de textes
- Analyse statistique des différences
- Détection des biais cryptographiques

### Génération de Rapports
- Visualisations interactives et explicatives
- Graphiques de distribution des différences
- Analyse comparative des probabilités
- Explications générées par IA pour différents niveaux d'expertise

## 📈 Visualisations

Le projet génère deux types principaux de visualisations :

1. **Distribution des Différences** :
   - Montre la répartition des différences en sortie
   - Met en évidence les biais statistiques
   - Inclut des annotations explicatives

2. **Analyse des Probabilités** :
   - Compare les probabilités observées vs théoriques
   - Utilise une échelle logarithmique pour une meilleure lisibilité
   - Affiche le facteur de biais de l'attaque

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forker le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📜 Licence

[À définir]

## 📚 Références

1. Biham, E., & Shamir, A. (1991). Differential cryptanalysis of DES-like cryptosystems.
2. Feistel, H. (1973). Cryptography and Computer Privacy.
3. [Documentation OpenAI](https://platform.openai.com/docs/api-reference)
4. [Documentation Matplotlib](https://matplotlib.org/)
