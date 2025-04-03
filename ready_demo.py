import torch
import numpy as np
import heartpy as hp

class SuicideInclinationClassifier(torch.nn.Module):
    def __init__(self, diseases, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.models = {d: torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Sigmoid()).float() for d in diseases}
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizers = {d: torch.optim.Adam(self.models[d].parameters()) for d in list(self.models.keys())}
        self.mode = 'train'

    def forward(self, parameters: list, disease=None):
        if self.mode == 'train':
            probability = self.models[disease](parameters)

            label = [1, 0] if disease != None else [0, 1]

            loss = self.criterion(probability, torch.tensor(label).float())
            loss.backward()

            self.optimizers[disease].step()
            self.optimizers[disease].zero_grad()

            print(f'{disease} loss is {loss}')
        elif self.mode == 'test':
            probability = self.models[disease](parameters)

            return probability[0]

    def set_mode(self, mode):
        assert mode in ['train', 'test']

        self.mode = mode

        if self.mode == 'train':
            for d, m in self.models.items():
                m.train()
        elif self.mode == 'test':
            for d, m in self.models.items():
                m.eval()