import torch

import segmentation_models_pytorch as smp

from feedback_segmentation.losses import evidential_loss

from feedback_segmentation.layers.equilibrium import EquilibriumModel

from feedback_segmentation.layers.dirichlet import Dirichlet

from feedback_segmentation.layers.state import StateExpDecay

from torchvision.models.resnet import BasicBlock

class ResNetEncoder(smp.encoders.resnet.ResNetEncoder):
    """
    This class is a copy of the smp.econders.resnet.ResNetEncoder
    It is here to implement state layers in between blocks, however researched diverged from that
    """
    def __init__(self, out_channels, in_channels=3, depth=5, alpha=0.1, **kwargs):
        super().__init__(out_channels, depth=5, **kwargs)

        self.set_in_channels(in_channels)
        
    def forward(self, x):
        
        stages = self.get_stages()
        
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            
            features.append(x)
            
        return features

        
class UnetDecoder(smp.unet.decoder.UnetDecoder):
    """
    This class is a copy of the smp.unet.decoder.UnetDecoder
    It is here to implement state layers in between blocks, however researched diverged from that
    """
    def __init__(self, encoder_channels, decoder_channels, alpha=0.5, **kwargs):
        super(UnetDecoder, self).__init__(encoder_channels, decoder_channels, **kwargs)

    def forward(self, *features):

        features = features[1:] # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class UnetFPI(smp.unet.model.Unet, EquilibriumModel):
    """
    This class is a copy of the smp.unet.model.Unet
    It is here to implement state layers in between blocks, however researched diverged from that
    It inherits from EquilibriumModel which performs fixed point iteration
    """
    params={"out_channels": (3, 64, 64, 128, 256, 512),"block": BasicBlock,"layers": [3, 4, 6, 3],}
    def __init__(self, in_channels=3, class_filters=16, fb_filters=0, registers=0, classes=1, alpha=0.5,):
        
        super(UnetFPI, self).__init__(in_channels=in_channels+fb_filters, classes=classes)
        
        self.encoder=ResNetEncoder(in_channels=in_channels+fb_filters, alpha=alpha, **UnetFPI.params)
        self.decoder=UnetDecoder(encoder_channels=self.encoder.out_channels, alpha=alpha, decoder_channels=(256, 128, 64, 32, class_filters+fb_filters+registers))

        self.initialize()

        self.classes=classes

        self.state=torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def states(self):
        return {"h": self.state}

    def step(self, x):
        features=self.encoder(x,)

        decoder_output=self.decoder(*features,)

        return decoder_output
        
    def to(self, device="cuda:0", **kwargs):
        super(UnetFPI, self,).to(device=device, **kwargs)
        self.encoder.to(device=device, **kwargs)
        self.decoder.to(device=device, **kwargs)
        self.segmentation_head.to(device=device, **kwargs)
        if(self.classification_head is not None):
            self.classification_head.to(device=device, **kwargs)

    def forward(self, x, T=0, atol=1e-2):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        super(UnetFPI,self).equilibrium_nograph(x, T=T-1, atol=atol)

        return self.step(x)


class Unet(smp.Unet):

    UNetParams={"compute_error": True, "compute_attention": True}

    def __init__(self, shape, **params):

        super(Unet,self).__init__(in_channels=params["in_channels"], classes=params["classes"])

        self.compute_attention=params["compute_attention"]
        if(self.compute_attention):
            raise NotImplementedError
        
        self.compute_error=params["compute_error"]
        if(self.compute_error):
            self.error_head=StateExpDecay(params["classes"], shape[1:], tau=1e0)
            self.error_head.Q.requires_grad=True
            self.error_head.Qinv.requires_grad=True
            self.error_head.S.requires_grad=False
            self.ground_truth=None
            self.loss_fn=torch.nn.functional.cross_entropy

        self.activation=lambda x : x

        self.alpha=torch.nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.dt=torch.nn.Parameter(torch.tensor(1./6.), requires_grad=False)
        self.Tf=torch.nn.Parameter(torch.tensor(6.), requires_grad=False)

    def forward(self, x):
        if(self.compute_attention):
            x=self.attention(torch.cat((x, self.activation(self.tau)), dim=1))

        z=super(Unet, self,).forward(x)
        
        if(self.compute_error):
            z0=self.error_head(z, T=0)*(self.Tf*self.dt-0*self.dt)**(self.alpha-1)/torch.exp(torch.special.gammaln(self.alpha))
            if(self.training and self.ground_truth is not None): self.loss_fn(self.activation(z0), self.ground_truth).mean().backward(retain_graph=True)
    
            for t in range(1, 5):
                z0+=self.error_head(z, T=t)*(self.Tf*self.dt-t*self.dt)**(self.alpha-1)/torch.exp(torch.special.gammaln(self.alpha))
                if(self.training and self.ground_truth is not None): self.loss_fn(self.activation(z0), self.ground_truth).mean().backward(retain_graph=True)
    
            z0+=self.error_head(z, T=5)*(self.Tf*self.dt-5*self.dt)**(self.alpha-1)/torch.exp(torch.special.gammaln(self.alpha))
            return self.activation(z0)

        return z


class UnetDirichlet(Unet, Dirichlet):

    UNetParams={"compute_error": True, "compute_attention": True}

    def __init__(self, shape, **params):
        super(UnetDirichlet,self).__init__(shape, **params)

        if(self.compute_error):
            self.loss_fn=evidential_loss

        self.activation=super(UnetDirichlet, self).activation