from feistel import FeistelCipher
from differential import DifferentialAnalysis
import numpy as np

def example_f_function(right_half: int, round_key: int) -> int:
    """Fonction de tour F simple pour l'exemple."""
    return (right_half * round_key) & 0xFFFF  # S'assure que le résultat reste sur 16 bits

def main():
    # Paramètres du chiffrement
    BLOCK_SIZE = 32  # Taille totale du bloc en bits
    NUM_ROUNDS = 4   # Nombre de tours
    
    # Création d'une instance du chiffrement de Feistel
    cipher = FeistelCipher(
        rounds=NUM_ROUNDS,
        block_size=BLOCK_SIZE,
        f_function=example_f_function
    )
    
    # Génération de clés de tour aléatoires
    round_keys = [np.random.randint(0, 0xFFFF) for _ in range(NUM_ROUNDS)]
    
    # Message de test
    plaintext = 0x12345678
    print(f"\nMessage original: {plaintext:08x}")
    
    # Test du chiffrement
    ciphertext = cipher.encrypt(plaintext, round_keys)
    print(f"Message chiffré: {ciphertext:08x}")
    
    # Test du déchiffrement
    decrypted = cipher.decrypt(ciphertext, round_keys)
    print(f"Message déchiffré: {decrypted:08x}")
    
    # Analyse différentielle
    print("\nDémarrage de l'analyse différentielle...")
    analyzer = DifferentialAnalysis(cipher, BLOCK_SIZE)
    
    # Trouve la meilleure caractéristique différentielle
    num_pairs = 1000
    input_diff, output_diff, probability = analyzer.find_best_differential(
        num_pairs=num_pairs,
        round_keys=round_keys
    )
    
    print(f"\nRésultats de l'analyse différentielle:")
    print(f"Différence en entrée: {input_diff:08x}")
    print(f"Différence en sortie: {output_diff:08x}")
    print(f"Probabilité: {probability:.2%}")

if __name__ == "__main__":
    main()
