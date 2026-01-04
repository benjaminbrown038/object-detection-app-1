import torch
import torch.nn as nn

class TinyYOLO(nn.Module):
    def __init__(self, num_classes=1, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Output: [x, y, w, h, conf, class]
        self.head = nn.Conv2d(128, (5 + num_classes), 1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
