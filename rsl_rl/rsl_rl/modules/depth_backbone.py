import torch
import torch.nn as nn
import sys
import torchvision

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, prop_dim, output_dim) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone

        self.prop_mlp = nn.Sequential(
                                nn.Linear(prop_dim, 128),
                                activation,
                                nn.Linear(128, 32),
                                nn.Linear(32, 8),
                            )



        self.combination_mlp = nn.Sequential(
                                nn.Linear(32 + 8, 128),
                                activation,
                                nn.Linear(128, 32)
                            )

        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32),
                                nn.Linear(32, output_dim),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_latent = self.base_backbone(depth_image)
        prop_latent = self.prop_mlp(proprioception)
        combine_latent = self.combination_mlp(torch.cat((depth_latent, prop_latent), dim=-1))
        rnnlatent, self.hidden_states = self.rnn(combine_latent[:, None, :], self.hidden_states)
        estposition = self.output_mlp(rnnlatent.squeeze(1))
        
        return estposition

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]   1 2 60 106
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83] 32 56 102
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41] 32 28 51
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]  32 26 49
            nn.Linear(64 * 26 * 49, 128),
            activation,
            nn.Linear(128, 32)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        # images shape [n, 60, 106]
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent