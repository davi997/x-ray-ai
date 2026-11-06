"""تدريب نموذج تصنيف (ResNet18 Transfer Learning).
مثال:
python train.py --train data/chest/train --val data/chest/test --out models/chest.pt
python train.py --train data/brain/Training --val data/brain/Testing --out models/brain.pt
"""
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

p = argparse.ArgumentParser()
p.add_argument("--train", required=True); p.add_argument("--val", required=True)
p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=6)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(10),
                            transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), norm])
va_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
tr = datasets.ImageFolder(a.train, tr_tf); va = datasets.ImageFolder(a.val, va_tf)
trl = DataLoader(tr, 32, shuffle=True, num_workers=2); val = DataLoader(va, 32)
print("Classes:", tr.classes)

# وزن للفئات لمعالجة عدم توازن الداتا
counts = torch.bincount(torch.tensor(tr.targets)).float()
w = (counts.sum() / counts).to(dev)

m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
m.fc = nn.Linear(512, len(tr.classes)); m = m.to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-4); lossf = nn.CrossEntropyLoss(weight=w)
best = 0
for ep in range(a.epochs):
    m.train()
    for x, y in trl:
        x, y = x.to(dev), y.to(dev); opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
    m.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in val:
            ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    acc = ok / n; print(f"epoch {ep+1}: val acc {acc:.3f}")
    if acc >= best:
        best = acc
        torch.save({"state": m.state_dict(), "classes": tr.classes}, a.out)
print("Best val acc:", best)
