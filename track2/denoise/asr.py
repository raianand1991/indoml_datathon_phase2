"""Frozen SraVaani transcription, pinned to one commit."""
import torch


def _text(h):
    if isinstance(h, str):
        return h.strip()
    if isinstance(h, dict) and isinstance(h.get("text"), str):
        return h["text"].strip()
    if hasattr(h, "text") and isinstance(h.text, str):
        return h.text.strip()
    if isinstance(h, (list, tuple)) and len(h) == 1:
        return _text(h[0])
    raise TypeError(f"unrecognised ASR hypothesis: {type(h)}")


class SraVaani:
    REPO = "ARTPARK-IISc/SraVaani-1.0"

    def __init__(self, revision, device="cpu", token=None):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.REPO, revision=revision, code_revision=revision,
            trust_remote_code=True, token=token).to(device).eval()

    def __call__(self, wav_path):
        # one file per call keeps the file -> text mapping unambiguous;
        # inference_mode is avoided because the remote TorchScript code
        # trips a version-counter error under it
        with torch.inference_mode(False), torch.no_grad():
            hyp = self.model.transcribe([str(wav_path)], return_hypotheses=True)
        return _text(hyp)
