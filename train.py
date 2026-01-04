import torch
from torch.utils.data import DataLoader
from models.tiny_yolo import TinyYOLO
from data.synthetic_dataset import SyntheticRectDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SyntheticRectDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = TinyYOLO().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(5):
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs).mean(dim=[2,3])

        loss = loss_fn(preds, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")

torch.save(model.state_dict(), "tiny_yolo.pt")
