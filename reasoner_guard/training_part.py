import torch.nn as nn
import torch
    
# ======================== Prompt Embedding Module ========================
class PromptEmbeddingModule(torch.nn.Module):
    def __init__(self, init_prompt):
        super().__init__()
        self.prompt_embeds = torch.nn.Parameter(init_prompt)

    def forward(self):
        return self.prompt_embeds