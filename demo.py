import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me! DONE
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        self.l1 = nn.Linear(5, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 400)
        self.out = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)
        return x


def main():
    """Train the model and save the checkpoint"""

    # Create model
    model = PutNet()

    # Load dataset
    df = pd.read_csv("bs-put-1k.csv")
    # Set up training
    x = torch.Tensor(df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(df[["value"]].to_numpy())

    criterion = nn.MSELoss()
    ## optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1000
    data_size = len(x)
    batch_size = 256 #1024
    train_size = int(0.8 * data_size)
    x_tr = x[:train_size]
    # y_hat = model(x_tr)
    y_tr = y[:train_size]
    for i in range(epochs):

        # TODO: Modify to account for dataset size
        perm = torch.randperm(train_size)
        for j in range(0, train_size, batch_size):
            indices = perm[j: min(j+batch_size, train_size)]
            x_tr_batch, y_tr_batch = x_tr[indices], y_tr[indices]
            y_hat_batch = model(x_tr_batch)
            # Calculate training loss
            training_loss = criterion(y_hat_batch, y_tr_batch)

            # Take a step
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        # Check validation loss
        with torch.no_grad():
            # TODO: use a proper validation set
            validation_loss = criterion(model(x[train_size+1:]), y[train_size+1:])  # the rest 20% as val

        print(f"Iteration: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} ")

    torch.save(model.state_dict(), "simple-model.pt")


if __name__ == "__main__":
    main()
