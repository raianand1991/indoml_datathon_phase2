"""Training objective used for the final distillation stage."""
import torch

W_NOISE_SISDR = 0.30
W_COMPLEX_NOISE = 0.25
W_NOISE_WAVE = 0.10
W_OUTSIDE_IDENTITY = 0.20


def neg_si_sdr(y, s):
    y = y - y.mean(-1, keepdim=True)
    s = s - s.mean(-1, keepdim=True)
    target = (y * s).sum(-1, keepdim=True) / (s.square().sum(-1, keepdim=True) + 1e-8) * s
    err = y - target
    return -10 * torch.log10((target.square().sum(-1) + 1e-8) / (err.square().sum(-1) + 1e-8)) / 20


def complex_stft_loss(pred, target, sizes=(256, 512, 1024)):
    total = pred.new_zeros(())
    for n in sizes:
        win = torch.hann_window(n, device=pred.device)
        P = torch.stft(pred.float(), n, hop_length=n // 4, window=win, return_complex=True, pad_mode="constant")
        T = torch.stft(target.float(), n, hop_length=n // 4, window=win, return_complex=True, pad_mode="constant")
        total = total + (P - T).abs().mean() / T.abs().mean().clamp_min(1e-3)
    return total / len(sizes)


def distillation_loss(model, x, s, n, event_mask):
    """x = s + n. event_mask marks samples where the target noise is active.

    speech SI-SDR + noise SI-SDR + multi-resolution complex STFT on the noise
    + normalised L1 on the noise + an identity term that penalises changing the
    input outside noise events. The mask is used only here, never at inference.
    """
    y, pred_n, strength, _ = model(x[:, None], return_aux=True)
    y, pred_n = y[:, 0], pred_n[:, 0]

    speech = neg_si_sdr(y, s).mean()
    has_noise = n.square().mean(-1) > 1e-8
    noise_si = neg_si_sdr(pred_n[has_noise], n[has_noise]).mean() if has_noise.any() else y.new_zeros(())
    cplx = complex_stft_loss(pred_n, n)
    scale = n.square().mean(-1).sqrt().clamp_min(0.01)[:, None]
    noise_l1 = ((pred_n - n).abs() / scale).mean()
    outside = 1 - event_mask
    s_scale = s.square().mean(-1).sqrt().clamp_min(0.01)[:, None]
    identity = (((y - x).abs() / s_scale) * outside).sum() / outside.sum().clamp_min(1)

    loss = (speech + W_NOISE_SISDR * noise_si + W_COMPLEX_NOISE * cplx
            + W_NOISE_WAVE * noise_l1 + W_OUTSIDE_IDENTITY * identity)
    return loss, dict(speech_si=speech, noise_si=noise_si, complex_noise=cplx,
                      noise_wave=noise_l1, outside_identity=identity,
                      mean_strength=strength.mean())
