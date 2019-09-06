import matplotlib.pyplot as plt
import seaborn as sns

import torch
import os
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

batch_size = 64
epochs = 20
no_cuda = False
seed = 1
log_interval = 50

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if cuda else "cpu")
print(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# train_root = 'Data/fruits-360/Training'
# val_root = 'Data/fruits-360/Test'
train_root = 'frames/test'
val_root = 'frames/test'

# train_loader_food = torch.utils.data.DataLoader(
#     datasets.ImageFolder(train_root, transform=transforms.ToTensor()),
#     batch_size = batch_size, shuffle=True, **kwargs)

# val_loader_food = torch.utils.data.DataLoader(
#     datasets.ImageFolder(val_root, transform=transforms.ToTensor()),
#     batch_size = batch_size, shuffle=True, **kwargs)
image_width = 220
image_heigh = 160


train_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, \
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_width, image_heigh)), transforms.ToTensor(),  transforms.Normalize([0.5], [0.5]) ])), 
    batch_size = batch_size, shuffle=True, **kwargs)


val_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_root,  \
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_width, image_heigh)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])), 
    batch_size = batch_size, shuffle=True, **kwargs)

class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        def conv2d_size_out(size,  stride, kernel_size=3, padding = 1):
            return (size - (kernel_size) + 2 * padding) // stride + 1
            
        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(image_width, stride=1), stride=2), stride=1), stride=2)
        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(image_heigh, stride=1), stride=2), stride=1), stride=2)

        print(self.convw, self.convh)
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(self.convw * self.convh * 16, 2048)
        self.fc_bn1 = nn.BatchNorm1d(2048)
        self.fc21 = nn.Linear(2048, 2048)
        self.fc22 = nn.Linear(2048, 2048)

        # Sampling vector
        self.fc3 = nn.Linear(2048, 2048)
        self.fc_bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, self.convw * self.convh * 16)
        self.fc_bn4 = nn.BatchNorm1d(self.convw * self.convh * 16)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, self.convw * self.convh * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, self.convw, self.convh)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, 1, image_width, image_heigh)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

model = VAE_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


loss_mse = customLoss()

val_losses = []
train_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader_food):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader_food.dataset)))
    train_losses.append(train_loss / len(train_loader_food.dataset))



def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader_food):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_mse(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 1, image_width, image_heigh)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 2048).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, image_width, image_heigh),
                   'results/sample_' + str(epoch) + '.png')

plt.figure(figsize=(15,10))
plt.plot(range(len(train_losses)),train_losses)
plt.plot(range(len(val_losses)),val_losses)
plt.title("Validation loss and loss per epoch",fontsize=18)
plt.xlabel("epoch",fontsize=18)
plt.ylabel("loss",fontsize=18)
plt.legend(['Training Loss','Validation Loss'],fontsize=14)
plt.show()