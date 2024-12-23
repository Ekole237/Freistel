from typing import Dict, List, Tuple
import numpy as np
from collections import Counter

class DifferentialAnalysis:
    def __init__(self, cipher, block_size: int):
        """
        Initialise l'analyse différentielle.
        
        Args:
            cipher: Instance du chiffrement à analyser
            block_size (int): Taille du bloc en bits
        """
        self.cipher = cipher
        self.block_size = block_size
        self.mask = (1 << block_size) - 1
        
    def generate_pair(self, plaintext: int, difference: int) -> Tuple[int, int]:
        """
        Génère une paire de textes clairs avec une différence donnée.
        
        Args:
            plaintext (int): Premier texte clair
            difference (int): Différence souhaitée
            
        Returns:
            Tuple[int, int]: Paire de textes clairs
        """
        return plaintext, plaintext ^ difference
        
    def collect_differences(self, num_pairs: int, input_diff: int, 
                          round_keys: List[int]) -> Dict[int, int]:
        """
        Collecte les différences en sortie pour plusieurs paires.
        
        Args:
            num_pairs (int): Nombre de paires à analyser
            input_diff (int): Différence en entrée
            round_keys (List[int]): Clés de tour
            
        Returns:
            Dict[int, int]: Compteur des différences en sortie
        """
        differences = Counter()
        
        for _ in range(num_pairs):
            # Génère un texte clair aléatoire
            p1 = np.random.randint(0, 1 << self.block_size)
            p2 = p1 ^ input_diff
            
            # Chiffre les deux textes
            c1 = self.cipher.encrypt(p1, round_keys)
            c2 = self.cipher.encrypt(p2, round_keys)
            
            # Calcule la différence en sortie
            output_diff = c1 ^ c2
            differences[output_diff] += 1
            
        return differences
        
    def find_best_differential(self, num_pairs: int, round_keys: List[int]) -> Tuple[int, int, float]:
        """
        Trouve la meilleure caractéristique différentielle.
        
        Args:
            num_pairs (int): Nombre de paires à tester
            round_keys (List[int]): Clés de tour
            
        Returns:
            Tuple[int, int, float]: (différence entrée, différence sortie, probabilité)
        """
        best_prob = 0
        best_input_diff = 0
        best_output_diff = 0
        
        # Test des différences possibles (simplifié pour l'exemple)
        for input_diff in range(1, min(256, 1 << self.block_size)):
            differences = self.collect_differences(num_pairs, input_diff, round_keys)
            
            if differences:
                most_common_diff, count = differences.most_common(1)[0]
                prob = count / num_pairs
                
                if prob > best_prob:
                    best_prob = prob
                    best_input_diff = input_diff
                    best_output_diff = most_common_diff
                    
        return best_input_diff, best_output_diff, best_prob
