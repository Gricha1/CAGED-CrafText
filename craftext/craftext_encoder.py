# craftext_encode_model.py

import os
from enum import Enum
from transformers import AutoModel, AutoTokenizer
import torch

os.environ['HF_HOME'] = "."
    
class EncodeForm(Enum):
    EMBEDDING = "embedding"
    TOKEN = "token"

class EncodeModel:
    def __init__(self, model_name="distilbert-base-uncased", form_to_use=EncodeForm.EMBEDDING):
        """
        Initialize the EncodeModel with the specified model name and encoding form.
        """
        self.model_name = model_name
        self.form_to_use = form_to_use
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".")
        self.model = AutoModel.from_pretrained(model_name, cache_dir=".")

    def encode(self, instruction):
        """
        Encodes instructions based on form_to_use. Returns either tokens or embeddings.
        """
        if self.form_to_use == EncodeForm.TOKEN:
            return self.get_tokens(instruction)
        elif self.form_to_use == EncodeForm.EMBEDDING:
            return self.get_embeddings(instruction)
        
    def get_embeddings(self, instruction):
        """
        Generates embeddings for the given instruction.
        """
        inputs = self.tokenizer(instruction, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extract the CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.numpy()

    def get_tokens(self, instruction):
        """
        Generates tokens for the given instruction.
        """
        return self.tokenizer(instruction, padding=True, truncation=True, return_tensors='np')['input_ids']
