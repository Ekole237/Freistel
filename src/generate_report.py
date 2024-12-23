from feistel import FeistelCipher
from differential import DifferentialAnalysis
from report_generator import ReportGenerator
import numpy as np
import json
from pathlib import Path

def generate_report_from_results():
    # Charger les résultats de l'analyse précédente
    results_file = Path("results/analysis_results.json")
    
    if not results_file.parent.exists():
        results_file.parent.mkdir(parents=True)
    
    # Exécuter l'analyse si nécessaire
    if not results_file.exists():
        # Paramètres du chiffrement
        BLOCK_SIZE = 32
        NUM_ROUNDS = 4
        
        # Création des instances
        cipher = FeistelCipher(
            rounds=NUM_ROUNDS,
            block_size=BLOCK_SIZE,
            f_function=lambda x, k: (x * k) & 0xFFFF
        )
        
        # Génération de clés de tour aléatoires
        round_keys = [np.random.randint(0, 0xFFFF) for _ in range(NUM_ROUNDS)]
        
        # Analyse différentielle
        analyzer = DifferentialAnalysis(cipher, BLOCK_SIZE)
        num_pairs = 1000
        input_diff, output_diff, probability = analyzer.find_best_differential(
            num_pairs=num_pairs,
            round_keys=round_keys
        )
        
        # Sauvegarder les résultats
        analysis_data = {
            "num_rounds": NUM_ROUNDS,
            "block_size": BLOCK_SIZE,
            "input_diff": input_diff,
            "output_diff": output_diff,
            "probability": float(probability),  # Convert numpy float to Python float
            "differences": {str(k): int(v) for k, v in analyzer.collect_differences(num_pairs, input_diff, round_keys).items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    else:
        # Charger les résultats existants
        with open(results_file, 'r') as f:
            analysis_data = json.load(f)
    
    # Générer le rapport
    print("\nGénération du rapport d'analyse...")
    report_gen = ReportGenerator()
    report_path = report_gen.generate_report(analysis_data)
    print(f"\nRapport généré : {report_path}")
    print(f"Visualisations sauvegardées dans le dossier : {report_path.parent}")

if __name__ == "__main__":
    generate_report_from_results()
