import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset


class ModelDataset(Dataset):
    def __init__(self, numpy_data_file_path, resize_image_size, transform=None):
        super(ModelDataset, self).__init__()
        self.transform = transform
        self.numpy_data = np.load(numpy_data_file_path) / 255.0
        self.reshaped_numpy_data = self.numpy_data.reshape(
            -1, 1, resize_image_size, resize_image_size
        )
        self.tensor_data = torch.from_numpy(self.reshaped_numpy_data).to(torch.float32)

    def __len__(self):
        return self.tensor_data.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.tensor_data[index])
        return self.tensor_data[index]


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self, input_channels, encoded_channels, loss_func, learning_rate, weight_decay
    ):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, encoded_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(encoded_channels),
            nn.ReLU(),
            nn.Conv2d(
                encoded_channels, encoded_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                encoded_channels,
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
        )
        self.criterion = loss_func
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x):
        encoded = self.encoder(x)
        loss = None
        if self.training:
            decoded = self.decoder(encoded)
            loss = self.criterion(decoded, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return encoded.detach(), loss.item() if loss is not None else 0

    def reconstruct(self, x):
        return self.decoder(x)



class StackedDenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels, loss_func, learning_rate, weight_decay):
        super(StackedDenoisingAutoencoder, self).__init__()
        self.dae_1 = DenoisingAutoencoder(
            input_channels, 32, loss_func, learning_rate, weight_decay
        )
        self.dae_2 = DenoisingAutoencoder(
            32, 64, loss_func, learning_rate, weight_decay
        )
        self.dae_3 = DenoisingAutoencoder(
            64, 128, loss_func, learning_rate, weight_decay
        )
        self.dae_4 = DenoisingAutoencoder(
            128, 256, loss_func, learning_rate, weight_decay
        )
        self.dae_5 = DenoisingAutoencoder(
            256, 512, loss_func, learning_rate, weight_decay
        )

    def forward(self, x):
        x, loss_dae_1 = self.dae_1(x)
        x, loss_dae_2 = self.dae_2(x)
        x, loss_dae_3 = self.dae_3(x)
        x, loss_dae_4 = self.dae_4(x)
        x, loss_dae_5 = self.dae_5(x)

        losses = [loss_dae_1, loss_dae_2, loss_dae_3, loss_dae_4, loss_dae_5]

        if self.training:
            return x, sum(losses)
        else:
            return x, self.reconstruct(x)

    def reconstruct(self, x):
        x_reconstruct = self.dae_5.reconstruct(x)
        x_reconstruct = self.dae_4.reconstruct(x_reconstruct)
        x_reconstruct = self.dae_3.reconstruct(x_reconstruct)
        x_reconstruct = self.dae_2.reconstruct(x_reconstruct)
        return self.dae_1.reconstruct(x_reconstruct)