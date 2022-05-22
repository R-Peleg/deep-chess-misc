from model import LastMovePredictor
from torch.utils.data import DataLoader
from position_dataset import LastMoveDataset
import torch
from torch import nn
import pytorch_lightning as pl 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    training_data = LastMoveDataset(open('data/2022-01.bare.[19999].pgn'))
    train_dataloader = DataLoader(training_data, batch_size=64)

    model = LastMovePredictor().to(device)
    print(model)

    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    main()
