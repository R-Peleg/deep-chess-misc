from model import LastMovePredictorModel
from torch.utils.data import DataLoader
from position_dataset import LastMoveDataset
import torch
from torch import nn

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    training_data = LastMoveDataset(open('data/2022-01.bare.[19999].pgn'))
    train_dataloader = DataLoader(training_data, batch_size=64)

    model = LastMovePredictorModel().to(device)
    print(model)
    # Prepare training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")


if __name__ == '__main__':
    main()
