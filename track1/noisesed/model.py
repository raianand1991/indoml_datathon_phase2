"""Three-head detector: log-mel CNN + frozen WavLM + frozen BEATs -> BiGRU.

Same architecture and front end as notebooks/01_detector_wavlm_beats.ipynb
(where the class is called BoundaryModel), which trained the checkpoint
`boundary_wavlm_beats_verified_stable_best.pt`. Needs torch, transformers,
librosa and the BEATs source from microsoft/unilm (beats/BEATs.py) with the
BEATs_iter3_plus_AS2M.pt checkpoint.

Outputs per 10 ms frame: noise logit, and onset / offset distance curves.
Training targets for the distance heads are |t - nearest boundary| clipped at
MAX_DIST seconds and divided by MAX_DIST, so 0 means "on a boundary".
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
HOP = 160
N_FFT = 512
N_MELS = 64
MAX_DIST = 0.5

_MEL_FB = None


def logmel(y):
    """Natural-log mel power, (N_MELS, T). Matches the training notebook."""
    global _MEL_FB
    import librosa
    if _MEL_FB is None:
        _MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)) ** 2
    return np.log(_MEL_FB @ spec + 1e-8).astype(np.float32)


def model_inputs(y):
    """Waveform -> (normalised mel, waveform padded to T * HOP, valid length)."""
    mel = logmel(y)
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)
    need = mel.shape[1] * HOP
    yy = np.pad(y, (0, max(0, need - len(y))))[:need].astype(np.float32)
    return mel.astype(np.float32), yy, min(len(y), need)


def _valid_mask(lengths, max_len, device):
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).long()


def _padding_mask(lengths, max_len, device):
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


class WavLMBranch(nn.Module):
    def __init__(self, name="microsoft/wavlm-base-plus"):
        super().__init__()
        from transformers import WavLMModel
        self.m = WavLMModel.from_pretrained(name)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.dim = int(self.m.config.hidden_size)
        self.m.eval()

    def forward(self, wav, n_frames, wav_lengths=None):
        self.m.eval()
        mask = None
        if wav_lengths is not None:
            mask = _valid_mask(wav_lengths.to(wav.device), wav.shape[1], wav.device)
        with torch.no_grad():
            h = self.m(wav, attention_mask=mask).last_hidden_state
        return F.interpolate(h.transpose(1, 2).float(), size=n_frames,
                             mode="linear", align_corners=False).transpose(1, 2)


class BEATsBranch(nn.Module):
    def __init__(self, model, dim):
        super().__init__()
        self.m, self.dim = model, int(dim)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def forward(self, wav, n_frames, wav_lengths=None):
        self.m.eval()
        pm = None
        if wav_lengths is not None:
            pm = _padding_mask(wav_lengths.to(wav.device), wav.shape[1], wav.device)
        with torch.no_grad():
            h = self.m.extract_features(wav.float(), padding_mask=pm)[0]
        return F.interpolate(h.transpose(1, 2).float(), size=n_frames,
                             mode="linear", align_corners=False).transpose(1, 2)


class StableBoundaryModel(nn.Module):
    def __init__(self, beats_model, beats_dim, n_mels=N_MELS, ch=(32, 64, 128, 128),
                 rnn=192, wavlm_proj=128, beats_proj=128):
        super().__init__()
        blocks, cin = [], 1
        for c in ch:
            # pool over frequency only, so the 10 ms time grid is kept
            blocks += [nn.Conv2d(cin, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.AvgPool2d((2, 1)), nn.Dropout2d(0.1)]
            cin = c
        self.cnn = nn.Sequential(*blocks)
        self.mel_feat = cin * (n_mels // (2 ** len(ch)))

        self.wavlm = WavLMBranch()
        self.wavlm_proj = nn.Sequential(nn.Linear(self.wavlm.dim, wavlm_proj),
                                        nn.LayerNorm(wavlm_proj), nn.ReLU())
        self.beats = BEATsBranch(beats_model, beats_dim)
        self.beats_proj = nn.Sequential(nn.Linear(beats_dim, beats_proj),
                                        nn.LayerNorm(beats_proj), nn.ReLU())

        self.rnn = nn.GRU(self.mel_feat + wavlm_proj + beats_proj, rnn, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)
        self.noise_head = nn.Linear(2 * rnn, 1)
        self.onset_head = nn.Linear(2 * rnn, 1)
        self.offset_head = nn.Linear(2 * rnn, 1)

    def forward(self, x_mel, wav, wav_lengths=None):
        """x_mel: (B, 1, n_mels, T); wav: (B, samples). Returns three (B, T) tensors."""
        with torch.amp.autocast(device_type=x_mel.device.type, enabled=False):
            hm = self.cnn(x_mel.float())
            b, c, fq, t = hm.shape
            hm = hm.permute(0, 3, 1, 2).reshape(b, t, c * fq)
            hw = self.wavlm(wav.float(), t, wav_lengths)
            hb = self.beats(wav.float(), t, wav_lengths)
            h = torch.cat([hm, self.wavlm_proj(hw), self.beats_proj(hb)], dim=-1)
            h, _ = self.rnn(h)
            noise_logit = self.noise_head(h).squeeze(-1)
            onset = torch.sigmoid(self.onset_head(h).squeeze(-1))
            offset = torch.sigmoid(self.offset_head(h).squeeze(-1))
        return noise_logit, onset, offset


BoundaryModel = StableBoundaryModel  # name used in the training notebook


def load_beats(ckpt_path, beats_src_dir):
    """BEATs from the unilm source tree and the iter3+ AS2M checkpoint."""
    import sys
    if beats_src_dir not in sys.path:
        sys.path.insert(0, beats_src_dir)
    from BEATs import BEATs, BEATsConfig
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ck["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ck["model"], strict=True)
    return model.eval(), int(cfg.encoder_embed_dim)


def load_stable(ckpt_path, beats_ckpt, beats_src_dir, device="cpu"):
    beats, dim = load_beats(beats_ckpt, beats_src_dir)
    model = StableBoundaryModel(beats, dim)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # encoder weights come from the pretrained checkpoints, not the saved state
    bad = [k for k in missing if not k.startswith(("wavlm.m.", "beats.m."))]
    if bad or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {bad[:10]} unexpected {unexpected[:10]}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
