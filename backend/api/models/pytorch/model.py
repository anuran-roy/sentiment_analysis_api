from torch import nn
import torch.functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


class Word2VecCBOWModel(nn.Module):
    """This model learns word embeddings from the Twitter corpus using the CBOW technique."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        *args: Optional[List[Any]],
        **kwargs: Optional[Dict[str, Any]]
    ):
        super(Word2VecCBOWModel, self).__init__(*args, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.neural_network = nn.Linear(self.embedding_size, self.vocabulary_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(input_tensor)
        embedding = embedding.mean(axis=1)
        embedding = self.neural_network(embedding)
        return embedding


class Word2VecSkipgramModel(nn.Module):
    """This model learns word embeddings from the Twitter corpus using the Skipgram technique."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        *args: Optional[List[Any]],
        **kwargs: Optional[Dict[str, Any]]
    ):
        super(Word2VecSkipgramModel, self).__init__(*args, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.neural_network = nn.Linear(self.embedding_size, self.vocabulary_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(input_tensor)
        embedding = self.neural_network(embedding)
        return embedding
