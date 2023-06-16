import sys

sys.path.append("/content/Human-Segmentation-Dataset-master")
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from torch.utils.data import Dataset
import helper
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# Configurações

CSV_FILE = "/Human-Segmentation-Dataset-master/train.csv"  # Caminho para o arquivo CSV contendo informações sobre as imagens e máscaras.
DATA_DIR = "/Image_Segmentation/"  # Diretório onde as imagens e máscaras estão armazenadas.

DEVICE = "CUDA"  # Dispositivo de execução (por exemplo, "cuda" para usar a GPU).

EPOCHS = 25  # Número de épocas de treinamento.
LR = 0.003  # Taxa de aprendizado.
IMG_SIZE = 320  # Tamanho das imagens.

ENCODER = "time-efficientnet-b0"  # Modelo de codificação a ser usado.
WEIGHTS = "imagenet"  # Pesos pré-treinados para o modelo de codificação.

# Leitura do arquivo CSV contendo informações sobre as imagens e máscaras
df = pd.read_csv(CSV_FILE)
df.head()

# Seleção de uma linha do DataFrame para visualização
row = df.iloc[4]

image_path = row.images
mask_path = row.masks

# Carregamento da imagem e máscara usando OpenCV
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

# Exibição da imagem e máscara usando matplotlib
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title("IMAGE")
ax1.imshow(image)

ax2.set_title("GROUND TRUTH")
ax2.imshow(mask, cmap="gray")

# Divisão dos dados em conjunto de treinamento e conjunto de validação
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Funções de aumento de dados para treinamento e validação
def get_train_augs():
    return A.Compose(
        [A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    )

def get_valid_augs():
    return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

# Classe do conjunto de dados personalizado
class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row.images
        mask_path = row.masks

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask

# Criação dos conjuntos de dados de treinamento e validação
trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

# Exemplo de visualização de uma imagem e máscara do conjunto de treinamento
idx = 3
image, mask = trainset[idx]
helper.show_image(image, mask)

# Carregamento dos conjuntos de dados em lotes usando DataLoader
BATCH_SIZE = 32
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

print(f"total no. of batches in trainloader : {len(trainloader)}")
print(f"total no. of batches in validloader : {len(validloader)}")

for image, mask in trainloader:
    break

print(f"One batch image shape: {image.shape}")
print(f"One batch mask shape : {mask.shape}")

# Criação do modelo de segmentação
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.arc = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)

        if masks is not None:
            loss1 = DiceLoss(mode="binary")(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2

        return logits

model = SegmentationModel()
model.to(DEVICE)

# Função de treinamento
def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Função de avaliação
def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, loss = model(images, masks)

            total_loss += loss.item()

    return total_loss / len(data_loader)

# Treinamento do modelo
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_valid_loss = np.Inf

for i in range(EPOCHS):
    train_loss = train_fn(trainloader, model, optimizer)
    valid_loss = eval_fn(validloader, model)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), "best_model.pt")
        print("SAVED-MODEL")
        best_valid_loss = valid_loss

    print(f"Epoch : {i+1} Train_loss : {train_loss} Valid_loss : {valid_loss}")

# Inferência
idx = 20
model.load_state_dict(torch.load("/content/best_model.pt"))

image, mask = validset[idx]
logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0

helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))
