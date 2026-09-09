from __future__ import annotations

import math

import torch
from torch import nn


class UnetSkipConnectionBlock(nn.Module):
    """Legacy pix2pix block retained for checkpoint compatibility."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        *,
        outermost: bool = False,
        innermost: bool = False,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        input_nc = outer_nc if input_nc is None else input_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            if submodule is None:
                raise ValueError("outermost block requires a submodule")
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            layers = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            layers = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            if submodule is None:
                raise ValueError("intermediate block requires a submodule")
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            layers = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, value):
        output = self.model(value)
        return output if self.outermost else torch.cat([value, output], dim=1)


class FiLM(nn.Module):
    def __init__(self, num_features: int, cond_dim: int = 1, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, num_features * 2),
        )

    def forward(self, value, condition):
        gamma, beta = self.net(condition).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return value * (1.0 + gamma) + beta


class FiLMUnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        *,
        outermost: bool = False,
        innermost: bool = False,
        use_dropout: bool = False,
        use_film_here: bool = False,
        cond_dim: int = 1,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule
        input_nc = outer_nc if input_nc is None else input_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        self.film = FiLM(inner_nc, cond_dim=cond_dim) if use_film_here else None

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            up_layers: list[nn.Module] = [uprelu, upconv, upnorm]
            if use_dropout:
                up_layers.append(nn.Dropout(0.5))
            self.up = nn.Sequential(*up_layers)

    def forward(self, value, condition=None):
        down = self.down(value)
        if self.innermost:
            if self.film is not None:
                if condition is None:
                    raise ValueError("FiLM checkpoint requires a condition tensor")
                down = self.film(down, condition)
            up = self.up(down)
        else:
            if self.submodule is None:
                raise RuntimeError("non-innermost FiLM block is missing its submodule")
            up = self.up(self.submodule(down, condition))
        return up if self.outermost else torch.cat([value, up], dim=1)


class UnetGenerator(nn.Module):
    """pix2pix U-Net with optional FiLM conditioning.

    Attribute names intentionally match the provided training package so its
    state dict can be loaded without renaming keys.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 256,
        ngf: int = 64,
        use_dropout: bool = True,
        use_film: bool = False,
        cond_dim: int = 1,
    ) -> None:
        super().__init__()
        num_downs = round(math.log2(image_size))
        if 2**num_downs != image_size:
            raise ValueError("image_size must be a power of two")
        self.use_film = use_film

        def make_block(
            outer_nc: int,
            inner_nc: int,
            input_nc: int | None = None,
            submodule: nn.Module | None = None,
            *,
            outermost: bool = False,
            innermost: bool = False,
            use_dropout_here: bool = False,
            film_here: bool = False,
        ) -> nn.Module:
            if use_film:
                return FiLMUnetSkipConnectionBlock(
                    outer_nc,
                    inner_nc,
                    input_nc=input_nc,
                    submodule=submodule,
                    outermost=outermost,
                    innermost=innermost,
                    use_dropout=use_dropout_here,
                    use_film_here=film_here,
                    cond_dim=cond_dim,
                )
            return UnetSkipConnectionBlock(
                outer_nc,
                inner_nc,
                input_nc=input_nc,
                submodule=submodule,
                outermost=outermost,
                innermost=innermost,
                use_dropout=use_dropout_here,
            )

        block = make_block(
            ngf * 8, ngf * 8, innermost=True, film_here=use_film
        )
        for _ in range(num_downs - 5):
            block = make_block(
                ngf * 8,
                ngf * 8,
                submodule=block,
                use_dropout_here=use_dropout,
            )
        block = make_block(ngf * 4, ngf * 8, submodule=block)
        block = make_block(ngf * 2, ngf * 4, submodule=block)
        block = make_block(ngf, ngf * 2, submodule=block)
        self.model = make_block(
            out_channels,
            ngf,
            input_nc=in_channels,
            submodule=block,
            outermost=True,
        )

    def forward(self, value, condition=None):
        if self.use_film:
            return self.model(value, condition)
        return self.model(value)
