import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(
        self, texts: Union[str, List[str]], convert_to_numpy: bool = True
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=convert_to_numpy)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding."""
        return self.encode([text])[0]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        test_embedding = self.encode(["test"])
        return test_embedding.shape[1]
