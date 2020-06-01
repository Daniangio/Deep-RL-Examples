import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0):
        super(MyModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    net = MyModule(num_inputs=2, num_classes=3)
    data = torch.FloatTensor([[2, 3]])
    for batch_x, batch_y in iterate_batches(data, batch_size=32):
        batch_x_t = torch.tensor(batch_x)
        batch_y_t = torch.tensor(batch_y)
        out_t = net(batch_x_t)
        loss_t = loss_function(out_t, batch_y_t)
        loss_t.backward() # Calcola i gradienti sull'intero grafico
        optimizer.step() # Applica l'ottimizzazione (cambia i pesi in base ai gradienti e al learning rate)
        optimizer.zero_grad() # Resetta i gradienti