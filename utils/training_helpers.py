from utils.timer import Timer
from utils.plot import RealTimeVisualizer
import torch

def train_cnn(net, device, train_loader, test_loader, optimizer, criterion, n_epochs):
    timer = Timer()
    visualizer = RealTimeVisualizer(xlabel="epoch", xlim=[0, n_epochs],
                                    legend=['train acc', 'train loss', 'test acc'])

    net.to(device)

    for epoch in range(n_epochs):
        timer.start()

        for i, data in enumerate(train_loader):
            X, y = data     # unpack tuple `data` into input tensors and their labels
            X, y = X.to(device), y.to(device) # Move data to GPU, if available
            optimizer.zero_grad()  # Reset gradient after each batch param-update

            y_hat_probas = net(X)  # Do forward pass for the current batch
            loss = criterion(y_hat_probas, y)  # Average loss across samples
            loss.backward()  # Do backward pass

            optimizer.step()  # Update model parameters

            # Plot train acc, train loss every 100 batches
            if i % 99 == 0:
                y_hat = torch.max(y_hat_probas, dim=1).indices
                train_acc = _get_accuracy(y, y_hat)
                train_loss = loss
                visualizer.add_point(x=epoch+i/len(train_loader),
                                     y=[train_acc, train_loss, None])
            # Free up GPU memory
            del X, y, y_hat_probas
            torch.cuda.empty_cache()

        test_acc = evaluate(net, test_loader, device)
        visualizer.add_point(x=epoch+1, y=[None, None, test_acc])

        timer.stop()

    total_training_time = timer.sum()
    print("Total training time: {}".format(total_training_time))
    print("train loss: {:.4f}".format(train_loss.item()), "\ntrain acc: {:.4f}".format(train_acc.item()),
          "\ntest acc: {:.4f}".format(test_acc.item()))

def evaluate(net, test_loader, device):
    labels = torch.tensor([], dtype=torch.int64).to(device)
    y_hats = torch.tensor([], dtype=torch.int64).to(device)

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_hat_probas = net(X)
        y_hat = torch.max(y_hat_probas, dim=1).indices

        # Append predictions and labels of the current minibatch
        labels = torch.cat([labels, y])
        y_hats = torch.cat([y_hats, y_hat])

        # Free up GPU memory
        del X, y, y_hat_probas, y_hat
        torch.cuda.empty_cache()

    return _get_accuracy(labels, y_hats)

def _get_accuracy(y, y_hat):
    assert isinstance(y, torch.Tensor) and isinstance(y_hat, torch.Tensor)
    assert y.shape == y_hat.shape
    # Return the total numbers of position where `y == y_hat`
    return (y==y_hat).sum()/float(y.shape[0])