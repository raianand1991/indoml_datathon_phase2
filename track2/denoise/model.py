"""Demucs dns64 with a clip-level noise conditioner.

Two additions on top of the stock Demucs forward pass:

* a spectral conditioner summarises the input's magnitude spectrum (mean and
  std over time of log1p(10|X|)) into a 128-d embedding;
* that embedding drives FiLM on the LSTM bottleneck and a scalar "removal
  strength" that scales the residual (input - speech estimate).

FiLM and strength layers are zero-initialised, so at the start of training the
model is numerically identical to the Demucs it was built from. Both branches
are bounded (tanh), so the conditioner can only nudge the base model.
"""
import torch
import torch.nn.functional as F
from torch import nn


class SpectralNoiseConditioner(nn.Module):
    def __init__(self, n_fft=512, hop=160, embedding=128):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        features = 2 * (n_fft // 2 + 1)
        self.net = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, 256), nn.SiLU(),
            nn.Linear(256, embedding), nn.SiLU(),
        )

    def forward(self, wave):
        with torch.amp.autocast(device_type=wave.device.type, enabled=False):
            spec = torch.stft(wave.float(), self.n_fft, hop_length=self.hop,
                              window=self.window.float(), return_complex=True,
                              pad_mode="constant").abs()
            feat = torch.log1p(10 * spec)
            feat = torch.cat([feat.mean(-1), feat.std(-1)], -1)
        return self.net(feat)


class NoiseAdaptiveResidualDemucs(nn.Module):
    def __init__(self, demucs, embedding=128, film_scale=0.10, strength_range=0.15):
        super().__init__()
        self.base = demucs
        self.conditioner = SpectralNoiseConditioner(embedding=embedding)
        bottleneck = int(self.base.encoder[-1][0].out_channels)
        self.film = nn.Linear(embedding, 2 * bottleneck)
        self.strength = nn.Linear(embedding, 1)
        for layer in (self.film, self.strength):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.film_scale = float(film_scale)
        self.strength_range = float(strength_range)

    def forward(self, mix, return_aux=False):
        from denoiser.resample import downsample2, upsample2

        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        original = mix
        emb = self.conditioner(original[:, 0])

        if self.base.normalize:
            std = original.mean(1, keepdim=True).std(-1, keepdim=True)
            x = original / (self.base.floor + std)
        else:
            std, x = 1.0, original
        length = x.shape[-1]
        x = F.pad(x, (0, self.base.valid_length(length) - length))
        if self.base.resample == 2:
            x = upsample2(x)
        elif self.base.resample == 4:
            x = upsample2(upsample2(x))

        skips = []
        for encode in self.base.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.base.lstm(x)
        x = x.permute(1, 2, 0)

        gamma, beta = self.film(emb).chunk(2, -1)
        x = x * (1 + self.film_scale * torch.tanh(gamma)[:, :, None]) \
            + self.film_scale * torch.tanh(beta)[:, :, None]

        for decode in self.base.decoder:
            skip = skips.pop(-1)
            x = decode(x + skip[..., :x.shape[-1]])
        if self.base.resample == 2:
            x = downsample2(x)
        elif self.base.resample == 4:
            x = downsample2(downsample2(x))

        speech = std * x[..., :length]
        strength = 1 + self.strength_range * torch.tanh(self.strength(emb))[:, :, None]
        residual = original - speech
        noise = strength * residual
        enhanced = speech + (1 - strength) * residual   # == original - noise
        if return_aux:
            return enhanced, noise, strength, emb
        return enhanced


def build(checkpoint=None, device="cpu"):
    from denoiser import pretrained

    base = pretrained.dns64(pretrained=False)
    model = NoiseAdaptiveResidualDemucs(base)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for m in model.modules():
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()
    return model
