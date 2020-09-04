from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)

    def forward(self, sent):
        tokens = self.tokenizer.tokenize(sent)
        tokens = tokens[:512]
        input_ids = torch.tensor(self.tokenizer.encode(tokens, add_special_tokens=False, add_space_before_punct_symbol=True)).unsqueeze(0).to(device)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states, mems = outputs[:2]  # The last hidden-state is the first element of the output tuple
        # print(pooled.shape)
        return last_hidden_states