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
        # self.encode = nn.Sequential(
        #     nn.Conv2d(3, 16, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 2*z_dim, 1)
        # )
        self.decode = Decoder(image_size, latent_dim=z_dim))
        # self.decode = nn.Sequential(
        #     nn.Conv2d(z_dim, 256, 1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 64, 4),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 16, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 3, 4, 2, 1),
        #     nn.Sigmoid()
        # )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

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