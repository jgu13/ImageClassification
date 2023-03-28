from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        idx = x # BNC
        x = self.norm1(x)
        x = x.transpose(1, 2) # BNC -> BCN
        x = self.mlp_tokens(x) 
        x = x.transpose(1, 2) # BCN -> BNC
        x = x + idx # skip connection
        idx = x
        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = x + idx # skip connection
        return x
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim) # normalize over embeded number of channels
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        images = self.patchemb.forward(images) # images have size BNC
        # step 2 Go through the mixer blocks
        images = self.blocks.forward(images) # images have size BNC
        # step 3 go through layer norm
        images = self.norm(images)  # images have size BNC
        # step 4 Global averaging spatially
        images = images.mean(dim=1) # images have shape B x C
        # Classification
        classes = self.head(images) # classes have size B x num_classes
        return classes

    def get_weights(self):
        filters = {}
        first_block = self.blocks[0]
        mlp = first_block.mlp_tokens
        first_layer = mlp.fc1
        filters['fc1'] = first_layer.weight.reshape((-1, 8, 8))
        return filters

    def visualize(self, logdir):
        """ Visualize the token mixer layer in the desired directory """
        import matplotlib as mpl
        import os.path as osp
        filters = self.get_weights()
        first_fc_layer = filters['fc1']
        # clip weights to range between 0 and 1
        fmin, fmax = first_fc_layer.min(), first_fc_layer.max()
        first_fc_layer = (first_fc_layer - fmin) / (fmax - fmin)
        # print(first_fc_layer.shape)
        ncols = 8
        nrows = first_fc_layer.size()[0] // ncols if (first_fc_layer.size()[0] % ncols == 0) else 9
        fig, axs = mpl.pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
        axs = axs.ravel()
        for i in range(first_fc_layer.size()[0]):
            f = first_fc_layer[i,:,:].detach().cpu().numpy()
            # print(f.shape)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].imshow(f, cmap='gray')
        if logdir != None:
          mpl.pyplot.savefig(osp.join(logdir, 'visualize_first_fc_layer.png'))
        mpl.pyplot.show()
 
