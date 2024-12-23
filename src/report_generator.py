import os
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ReportGenerator:
    def __init__(self, output_dir: str = "results"):
        """
        Initialise le générateur de rapports.
        
        Args:
            output_dir (str): Répertoire de sortie pour les rapports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_visualizations(self, data: Dict[str, Any], filename_prefix: str):
        """
        Crée plusieurs visualisations des résultats.
        
        Args:
            data (Dict[str, Any]): Données à visualiser
            filename_prefix (str): Préfixe pour les noms de fichiers
        """
        # Configuration générale de matplotlib pour une meilleure lisibilité
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            plt.style.use('default')
            
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['figure.titlesize'] = 18

        # 1. Distribution des différences
        if 'differences' in data:
            plt.figure(figsize=(15, 8))
            differences = list(map(int, data['differences'].keys()))
            frequencies = list(data['differences'].values())
            
            # Création du graphique avec des couleurs attrayantes
            bars = plt.bar(range(len(differences)), frequencies, 
                          color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Ajout d'une ligne de tendance
            plt.plot(range(len(differences)), frequencies, 
                    color='red', linestyle='--', alpha=0.5, 
                    label='Tendance')
            
            plt.title('Distribution des Différences de Sortie\nFréquence d\'apparition de chaque différence', 
                     pad=20)
            plt.xlabel('Valeur de la Différence (en hexadécimal)')
            plt.ylabel('Nombre d\'Occurrences')
            
            # Amélioration des étiquettes
            plt.xticks(range(len(differences)), 
                      [f'0x{d:04x}' for d in differences], 
                      rotation=45, ha='right')
            
            # Ajout d'une grille pour faciliter la lecture
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Ajout d'annotations pour les valeurs importantes
            max_freq_idx = frequencies.index(max(frequencies))
            plt.annotate(f'Pic maximal\n{max(frequencies)} occurrences',
                        xy=(max_freq_idx, max(frequencies)),
                        xytext=(10, 10), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'))
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{filename_prefix}_distribution.png", dpi=300)
            plt.close()

        # 2. Probabilité par rapport au hasard
        plt.figure(figsize=(12, 8))
        prob = data['probability']
        random_prob = 1.0 / (1 << data['block_size'])
        probs = [random_prob, prob]
        labels = ['Probabilité\nAléatoire\n(Théorique)', 'Probabilité\nObservée\n(Attaque)']
        
        # Création des barres avec des couleurs significatives
        bars = plt.bar(labels, probs, 
                      color=['lightgray', 'lightcoral'],
                      edgecolor=['gray', 'darkred'],
                      alpha=0.7)
        
        plt.title('Comparaison des Probabilités\nEfficacité de l\'attaque vs Hasard', pad=20)
        plt.ylabel('Probabilité (échelle logarithmique)')
        plt.yscale('log')
        
        # Ajout des valeurs exactes sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        
        # Ajout d'une ligne horizontale pour la probabilité aléatoire
        plt.axhline(y=random_prob, color='red', linestyle='--', alpha=0.5,
                   label='Seuil de l\'aléatoire')
        
        # Calcul et affichage du facteur de biais
        bias_factor = prob / random_prob
        plt.text(0.5, prob/2,
                f'Facteur de biais : {bias_factor:.1f}x\nL\'attaque est {bias_factor:.1f} fois\nplus efficace que le hasard',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename_prefix}_probability.png", dpi=300)
        plt.close()

    def get_feistel_introduction(self) -> str:
        """Obtient une introduction sur les schémas de Feistel via OpenAI."""
        prompt = """Rédigez une introduction claire et concise sur les schémas de Feistel en cryptographie, incluant :
1. Leur principe de fonctionnement
2. Leurs caractéristiques principales
3. Leur importance dans la cryptographie moderne
4. Quelques exemples d'algorithmes connus utilisant ce schéma"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en cryptographie spécialisé dans les schémas de Feistel."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération de l'introduction: {str(e)}"

    def analyze_attack_results(self, data: Dict[str, Any]) -> str:
        """Analyse les résultats de l'attaque différentielle."""
        bias_factor = data['probability'] / (1.0 / (1 << data['block_size']))
        prompt = f"""Analysez en détail les résultats suivants d'une attaque différentielle sur un schéma de Feistel :

Configuration :
- Nombre de tours : {data['num_rounds']}
- Taille de bloc : {data['block_size']} bits

Résultats :
- Différence en entrée : {data['input_diff']:08x}
- Différence en sortie : {data['output_diff']:08x}
- Probabilité observée : {data['probability']:.2%}
- Probabilité aléatoire théorique : {1.0 / (1 << data['block_size']):.2e}

Fournissez :
1. Une explication détaillée de l'attaque différentielle réalisée
2. Une analyse des biais observés et leur signification
3. Une évaluation de la vulnérabilité du chiffrement
4. Des recommandations concrètes pour renforcer la sécurité"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en cryptanalyse spécialisé dans l'analyse des attaques différentielles."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de l'analyse des résultats: {str(e)}"

    def get_graph_explanations(self, data: Dict[str, Any]) -> str:
        """Génère des explications détaillées des graphiques pour les non-experts."""
        bias_factor = data['probability'] / (1.0 / (1 << data['block_size']))
        
        prompt = f"""Expliquez de manière simple et accessible les deux graphiques suivants d'une analyse cryptographique, en évitant le jargon technique autant que possible :

1. Le premier graphique (Distribution des Différences) :
   - Montre comment les différences se répartissent
   - Le pic maximal indique la différence la plus fréquente
   - Plus les pics sont marqués, plus l'attaque est efficace

2. Le second graphique (Comparaison des Probabilités) :
   - Compare la probabilité observée lors de l'attaque avec celle du hasard
   - Montre que l'attaque est {bias_factor:.1f} fois plus efficace que le hasard
   - Utilise une échelle logarithmique pour mieux visualiser les différences

Expliquez :
1. Ce que ces graphiques signifient pour un public non technique
2. Pourquoi ces résultats sont importants
3. Ce qu'ils nous apprennent sur la sécurité du système
4. Utilisez des analogies simples pour faciliter la compréhension"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en vulgarisation scientifique, spécialisé dans l'explication de concepts cryptographiques complexes à un public non technique."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération des explications: {str(e)}"

    def generate_report(self, data: Dict[str, Any]):
        """Génère un rapport complet."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"analysis_{timestamp}"
        
        # Créer les visualisations
        self.create_visualizations(data, filename_prefix)
        
        # Obtenir l'introduction, l'analyse et les explications des graphiques
        introduction = self.get_feistel_introduction()
        analysis = self.analyze_attack_results(data)
        graph_explanations = self.get_graph_explanations(data)
        
        report_content = f"""# Rapport d'Analyse Différentielle sur un Schéma de Feistel

## Introduction aux Schémas de Feistel
{introduction}

## Configuration de l'Expérimentation
- **Nombre de tours** : {data['num_rounds']}
- **Taille de bloc** : {data['block_size']} bits
- **Nombre de paires analysées** : {sum(data['differences'].values())}

## Description de l'Attaque Différentielle
L'attaque différentielle réalisée consiste à analyser la propagation des différences à travers le réseau de Feistel. 
Cette technique, introduite par Biham et Shamir, permet d'exploiter des biais statistiques dans la distribution des différences.

## Résultats de la Simulation

### Paramètres Observés
- **Différence en entrée** : 0x{data['input_diff']:08x}
- **Différence en sortie** : 0x{data['output_diff']:08x}
- **Probabilité observée** : {data['probability']:.2%}
- **Probabilité aléatoire théorique** : {1.0 / (1 << data['block_size']):.2e}

### Visualisation et Interprétation des Résultats

#### 1. Distribution des Différences
![Distribution des Différences]({filename_prefix}_distribution.png)

#### 2. Analyse des Probabilités
![Analyse des Probabilités]({filename_prefix}_probability.png)

### Explication des Graphiques pour les Non-Spécialistes
{graph_explanations}

## Analyse Technique Détaillée
{analysis}

## Conclusion
Cette analyse démontre l'importance d'une évaluation rigoureuse des schémas de chiffrement contre les attaques différentielles.
Les résultats obtenus permettent de mieux comprendre les forces et les faiblesses du schéma de Feistel étudié.

---
*Rapport généré le {datetime.now().strftime("%Y-%m-%d à %H:%M:%S")}*
"""
        
        report_path = self.output_dir / f"report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return report_path
