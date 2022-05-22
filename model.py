import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class LastMovePredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Flatten(),
            nn.Linear(64 * 64, 512),
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
        x = x.permute((0, 3, 1, 2))
        pred = self.predict(x)
        loss = F.cross_entropy(pred, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.permute((0, 3, 1, 2))
        pred = self.predict(x)
        loss = F.cross_entropy(pred, y)
        return loss
    