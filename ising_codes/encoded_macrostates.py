import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import minmax_scale

class VAE(nn.Module):

    def __init__(self, input_dim=900, hidden_dim=450, latent_dim=200):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

final_lattice_arr=[]
num_temp=200
temperatures = np.linspace(1.0, 5.0, num_temp)

for j in range (0,len(temperatures)):
    temp = str(temperatures[j])
    part2 = "_macrostates"
    ext = ".npy"
    combined_name = temp + part2 + ext
    final_arr=np.load(combined_name)
    final_lattice_arr.append(final_arr)

final_comb_array = np.concatenate(final_lattice_arr, axis=0)

final_comb_array = final_comb_array[99::100, ...]
final_comb_array = final_comb_array.reshape((len(final_comb_array), np.prod(final_comb_array.shape[1:])))
final_comb_array = minmax_scale(final_comb_array)

samples_per_temp=len(final_comb_array)/num_temp

loss=nn.MSELoss()
def vae_loss(x_recon, x, mu, log_var):
    recon_loss = loss(x_recon,x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence
tensor_data=torch.tensor(final_comb_array, dtype=torch.float32)
dataset = TensorDataset(tensor_data, tensor_data)  # VAE is unsupervised, so inputs are targets
train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
enc_loader = DataLoader(dataset, batch_size=100, shuffle=False)

vae = VAE(input_dim = 900, hidden_dim = 450, latent_dim = 200)
optimizer = Adam(vae.parameters(), lr=1e-3)
batch_size = 100
x_dim=900
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training loop
def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
             
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = vae_loss(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        
    return overall_loss
train(vae, optimizer, epochs=300, device=device)
def encode_inputs(vae, inputs):
    vae.eval()
    repr_array=[]
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(inputs):

            
            x_hat = vae.encoder(x)
            x_hat = x_hat.numpy()
            repr_array.append(x_hat.copy())
            

    final_repr = np.concatenate(repr_array, axis=0)
    return final_repr

result=[]
final_repr=encode_inputs(vae, enc_loader)

for j in range(0, final_repr.shape[0], int(samples_per_temp)):
    chunk = final_repr[j:j+int(samples_per_temp),:]

    averaged_chunk = np.mean(chunk, axis=0)
    averaged_chunk = averaged_chunk[np.newaxis, :]
    result.append(averaged_chunk.copy())
mean_repr=np.concatenate(result, axis=0)

np.save("representation_macrostates",mean_repr) 

