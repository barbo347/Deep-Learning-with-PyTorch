import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch import nn
from torchsummary import summary

torch.manual_seed(42)
from tqdm.notebook import tqdm

device = "cuda"  # image = image.to(device)

batch_size = 128  # trainloader, training loop

noise_dim = 64  # generator model

# optimizers parameters

lr = 0.0002
beta_1 = 0.5
beta_2 = 0.99

# Training variables

epochs = 20

train_augs = T.Compose(
    [T.RandomRotation((-20, +20)), T.ToTensor()]  # (h, w, c) -> (c, h, w)
)

trainset = datasets.MNIST("MNIST/", download=True, train=True, transform=train_augs)
image, label = trainset[9000]

plt.imshow(image.squeeze(), cmap="gray")

print("total images present in trainset are: ", len(trainset))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
print("Total no. of batches in trainloader: ", len(trainloader))

dataiter = iter(trainloader)

images, _ = next(iter(trainloader))

print(images.shape)

# 'show_tensor_images' : function is used to plot some of images from the batch


def show_tensor_images(tensor_img, num_images=16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()


show_tensor_images(images, num_images=16)


def get_disc_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block_1 = get_disc_block(1, 16, (3, 3), 2)
        self.block_2 = get_disc_block(16, 32, (5, 5), 2)
        self.block_3 = get_disc_block(32, 64, (5, 5), 2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=1)

    def forward(self, images):
        x1 = self.block_1(images)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)

        x4 = self.flatten(x3)
        x5 = self.linear(x4)

        return x5


D = Discriminator()
D.to(device)

summary(D, input_size=(1, 28, 28))


def get_gen_block(in_channels, out_channels, kernel_size, stride, final_block=False):
    if final_block == True:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.Tanh(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.block_1 = get_gen_block(noise_dim, 256, (3, 3), 2)
        self.block_2 = get_gen_block(256, 128, (4, 4), 1)
        self.block_3 = get_gen_block(128, 64, (3, 3), 2)

        self.block_4 = get_gen_block(64, 1, (4, 4), 2, final_block=True)

    def forward(self, r_noise_vec):
        # (bs, noise_dim) -> (bs, noise_dim, 1, 1)
        x = r_noise_vec.view(-1, self.noise_dim, 1, 1)

        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)

        return x4


G = Generator(noise_dim)
G.to(device)

summary(G, input_size=(1, noise_dim))

# Replace Random initialized weights to Normal weights


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


D = D.apply(weights_init)
G = G.apply(weights_init)


def real_loss(disc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = torch.ones_like(disc_pred)
    loss = criterion(disc_pred, ground_truth)
    return loss


def fake_loss(disc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = torch.zeros_like(disc_pred)
    loss = criterion(disc_pred, ground_truth)
    return loss


D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))

for i in range(epochs):
    total_d_loss = 0.0
    total_g_loss = 0.0

    for real_img, _ in tqdm(trainloader):
        real_img = real_img.to(device)
        noise = torch.randn(batch_size, noise_dim, device=device)

        # find loss and update weights for D

        D_opt.zero_grad()

        fake_img = G(noise)
        D_pred = D(fake_img)
        D_fake_loss = fake_loss(D_pred)

        D_pred = D(real_img)
        D_real_loss = real_loss(D_pred)

        D_loss = (D_fake_loss + D_real_loss) / 2

        total_d_loss += D_loss.item()

        D_loss.backward()
        D_opt.step()

        # find loss ad update weights for G

        G_opt.zero_grad()

        noise = torch.randn(batch_size, noise_dim, device=device)

        fake_img = G(noise)
        D_pred = D(fake_img)
        G_loss = real_loss(D_pred)

        total_g_loss += G_loss.item()

        G_loss.backward()
        G_opt.step()

avg_d_loss = total_d_loss / len(trainloader)
avg_g_loss = total_g_loss / len(trainloader)

print("Epoch : {} | D_loss : {} | G_loss : {}".format(i + 1, avg_d_loss, avg_g_loss))

show_tensor_images(fake_img)
