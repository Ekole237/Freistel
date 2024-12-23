from typing import Callable, Tuple
import numpy as np

class FeistelCipher:
    def __init__(self, rounds: int, block_size: int, f_function: Callable):
        """
        Initialise un chiffrement de Feistel.
        
        Args:
            rounds (int): Nombre de tours
            block_size (int): Taille du bloc en bits (doit être pair)
            f_function (Callable): Fonction de tour F
        """
        if block_size % 2 != 0:
            raise ValueError("La taille du bloc doit être paire")
            
        self.rounds = rounds
        self.block_size = block_size
        self.half_size = block_size // 2
        self.f = f_function
        
    def _split_block(self, block: int) -> Tuple[int, int]:
        """Divise un bloc en deux moitiés."""
        mask = (1 << self.half_size) - 1
        right = block & mask
        left = (block >> self.half_size) & mask
        return left, right
        
    def _combine_block(self, left: int, right: int) -> int:
        """Combine deux moitiés en un bloc."""
        return (left << self.half_size) | right
        
    def encrypt(self, plaintext: int, round_keys: list) -> int:
        """
        Chiffre un bloc en utilisant le schéma de Feistel.
        
        Args:
            plaintext (int): Bloc à chiffrer
            round_keys (list): Liste des clés de tour
            
        Returns:
            int: Bloc chiffré
        """
        if len(round_keys) != self.rounds:
            raise ValueError("Nombre incorrect de clés de tour")
            
        left, right = self._split_block(plaintext)
        
        for i in range(self.rounds):
            # Fonction de tour F avec la clé courante
            f_output = self.f(right, round_keys[i])
            # XOR et échange
            new_left = right
            new_right = left ^ f_output
            left, right = new_left, new_right
            
        # Pas d'échange final
        return self._combine_block(right, left)
        
    def decrypt(self, ciphertext: int, round_keys: list) -> int:
        """
        Déchiffre un bloc en utilisant le schéma de Feistel.
        
        Args:
            ciphertext (int): Bloc à déchiffrer
            round_keys (list): Liste des clés de tour
            
        Returns:
            int: Bloc déchiffré
        """
        # Utilise les clés dans l'ordre inverse
        return self.encrypt(ciphertext, round_keys[::-1])
