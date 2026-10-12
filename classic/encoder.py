from torch import nn

class Encoder(nn.modules):
    def __init__ (self, latent_dim = 64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 21, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLu(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(start_dim = 1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar