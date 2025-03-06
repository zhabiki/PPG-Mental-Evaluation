import torch
import numpy as np
import heartpy as hp

class SuicideInclinationClassifier(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = torch.nn.Conv1d(1, 1, 1260)
        self.tanh = torch.nn.Tanh()
        self.fc = torch.nn.Linear(4, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: np.ndarray, parameters: list):
        signal = torch.tensor(x).reshape(1, 76800)
        while(signal.shape != (1,1)):
            signal = self.conv(signal.float())
            signal = self.tanh(signal.float())

        info = torch.tensor([signal[0]] + parameters)

        logits = self.fc(info.float())
        probability = self.sigmoid(logits)

        return probability[0].item()