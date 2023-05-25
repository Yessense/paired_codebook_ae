from torch import nn

from paired_codebook_ae.model.factor_vae.encoder import Encoder
from paired_codebook_ae.model.factor_vae.decoder import Decoder
from .utils import kaiming_init, normal_init

class CLEVRFactorVAE(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, z_dim=10, image_size=[3, 128, 128]):
        super(CLEVRFactorVAE, self).__init__()
        self.z_dim = z_dim
        self.encode = Encoder(image_size, latent_dim=z_dim)
        self.decode = Decoder(image_size, latent_dim=z_dim)
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for module in self.encode.modules():
            initializer(module)
        for module in self.decode.modules():
            initializer(module)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()