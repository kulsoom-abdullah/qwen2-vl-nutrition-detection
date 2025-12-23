import torch
from transformers import ProcessorMixin
from unittest.mock import MagicMock

class MockProcessor(ProcessorMixin):
    """Mocks HuggingFace Processor for CPU-only testing."""
    def __init__(self, *args, **kwargs):
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        self.image_processor = MagicMock()
        
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "pixel_values": torch.randn(1, 3, 224, 224), 
            "attention_mask": torch.ones(1, 10)
        }
    
    def batch_decode(self, *args, **kwargs):
        return ["mock_decoded_string"]

    def apply_chat_template(self, *args, **kwargs):
        return "mock_prompt"
    
    def save_pretrained(self, *args, **kwargs):
        pass

class MockModel(torch.nn.Module):
    """Mocks HuggingFace Model for CPU-only testing."""
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.text_config.hidden_size = 1024
        self.config._name_or_path = "mock-model"
        self.config.name_or_path = "mock-model"
        # Dummy parameter to make it a valid module
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        self.tp_size = 1
        
    def forward(self, *args, **kwargs):
        # Return a mock output with loss
        return MagicMock(loss=torch.tensor(0.1))
    
    def get_input_embeddings(self):
        return torch.nn.Embedding(100, 1024)
    
    def get_output_embeddings(self):
        return torch.nn.Embedding(100, 1024)
