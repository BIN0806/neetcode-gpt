import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        unique_words = set()
        for pos in positive:
            for word in pos.split(" "):
                unique_words.add(word)
        for neg in negative:
            for word in neg.split(" "):
                unique_words.add(word)

        unique_words = sorted(unique_words)

        word_to_id = {word : i for i, word in zip( range(1, len(unique_words) + 1), unique_words)}
        
        
        positive = [ 
            torch.tensor([word_to_id[word] for word in pos.split(" ")])
            for pos in positive
        ]

        negative = [
            torch.tensor([word_to_id[word] for word in neg.split(" ")])
            for neg in negative    
        ]

        return torch.nn.utils.rnn.pad_sequence(positive + negative, padding_value=0, batch_first=True)