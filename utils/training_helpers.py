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
            X, y = data     #unpack tuple `data` into input tensors and their labels
            X, y = X.to(device), y.to(device) # Move data to GPU, if available
            optimizer.zero_grad()  # Reset các gradient trọng số sau mỗi lần cập nhật trọng số ở mỗi batch, nếu không chúng sẽ tích luỹ, gây ảnh hưởng đến các batch sau.

            y_hat = net(X)  # Thực hiện lượt truyền xuôi (forward pass) cho batch hiện tại -> có kết quả nhãn dự đoán
            loss = criterion(y_hat, y)  # Tính hàm mất mát dựa vào dự đoán và nhãn gốc
            loss.backward()  # Thực hiện lan truyền ngược (backward pass)

            optimizer.step()  # Cập nhật các tham số của mô hình

            # Plot train acc, train loss every 100 batches
            if i % 99 == 0:
                train_acc = _get_accuracy(y, y_hat)
                train_loss = loss.mean()
                visualizer.add_point(x=epoch+i/len(train_loader),
                                     y=[train_acc, train_loss, None])

        test_acc = evaluate(net, test_loader, device)
        visualizer.add_point(x=epoch+1, y=[None, None, test_acc])

        timer.stop()

    total_training_time = timer.sum()
    avg_training_time = timer.avg()
    print("Total training time: {}; on average: {} per epoch".format(total_training_time,
                                                                 avg_training_time))

def evaluate(net, test_loader, device):
    labels = torch.tensor([]).to(device)
    y_hats = torch.tensor([]).to(device)

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)

        # Append predictions and labels of the current minibatch
        labels = torch.cat([labels, y])
        y_hats = torch.cat([y_hats, y_hat])

    return _get_accuracy(labels, y_hats)

def _get_accuracy(y, y_hat):
    assert isinstance(y, torch.Tensor) and isinstance(y_hat, torch.Tensor)
    # Return the total numbers of position where `y == y_hat`
    return (y==y_hat).sum()/y.shape[0]