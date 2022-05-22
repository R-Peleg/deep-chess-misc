import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class LastMovePredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Softmax(0)
        )

    def forward(self, x):
        return self.predict(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        pred = self.predict(x)
        loss = F.cross_entropy(pred, y)
        return loss