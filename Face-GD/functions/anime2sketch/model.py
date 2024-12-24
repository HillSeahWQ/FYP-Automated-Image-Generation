import os
import torch 
import torch.nn as nn 
import functools
import torchvision


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for _ in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


def create_model():
    """Create a model for anime2sketch
    hardcoding the options for simplicity
    """
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    ckpt = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'netG.pth')) # changed from '/workspace/ddgm/functions/anime2sketch/netG.pth'
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    net.load_state_dict(ckpt)
    return net


class FaceSketchTool(nn.Module):
    def __init__(self, ref_path=None):
        super(FaceSketchTool, self).__init__()
        self.net = create_model().cuda()
        self.net.eval()
        
        self.to_tensor = torchvision.transforms.ToTensor()
        
        self.reference_img_path = "/workspace/ddgm/functions/anime2sketch/1397.png" if not ref_path else ref_path
        from PIL import Image
        img = Image.open(self.reference_img_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img
        
    def get_residual(self, image):
        sketch_image = self.net(image)
        sketch_ref = self.net(self.ref)
        
        return sketch_ref - sketch_image

    def get_residual_y(self, image, image_y):
        sketch_image = self.net(image)
        # sketch_ref = self.net(self.ref)
        sketch_ref = self.net(image_y)
        
        return sketch_ref - sketch_image
    
    def save_sketch(self, reference_img_path, output_path):
        from PIL import Image
        import numpy as np

        # preprocessing
        img = Image.open(reference_img_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()

        # Forward pass through the network to get the output (sketch)
        with torch.no_grad():  # Disable gradient computation
            output = self.net(img)  # Assuming the model outputs a tensor

        # Convert the output tensor back to a PIL image
        output = output.squeeze(0)  # Remove batch dimension
        output = output.cpu().clamp(-1, 1)  # Ensure values are in [-1, 1]
        
        # Convert to [0, 1] range for saving as image
        output = (output + 1) / 2  # Rescale to [0, 1]
        
        # If it's a color image (3 channels), make sure it's in HWC format
        output = output.permute(1, 2, 0).numpy()  # Convert to HWC format (height, width, channels)
        
        # Ensure the pixel values are in [0, 255] and convert to uint8
        output = (output * 255).astype(np.uint8)

        # Handle case if output is grayscale (single channel)
        if output.shape[2] == 1:  # Single channel, grayscale image
            output = output.squeeze(axis=-1)  # Remove channel dimension for grayscale

        # Save the image
        output_image = Image.fromarray(output)  # This should work now
        output_image.save(output_path)
        print(f"Saved sketch image to {output_path}")

    def get_gaussian_kernel(self, image, sigma):
        sketch_image = self.net(image)
        sketch_ref = self.net(self.ref)

        distance = torch.norm(sketch_image - sketch_ref, dim=-1)

        # Apply Gaussian kernel
        gaussian_similarity = torch.exp(-distance**2 / (2 * sigma**2))

        return gaussian_similarity