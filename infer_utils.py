import torch
import torchaudio
import librosa

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style_from_dicts(model, ref_dicts, device='cuda'):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)
    
    return reference_embeddings

def compute_style_from_path(model, path, device='cuda'):

    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1)) # 1X128
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1)) # 1X128

    ref_sp = torch.cat([ref_s, ref_p], dim=1) # 1X256

    return ref_sp

def compute_style_from_two_paths(model, style_path, predictor_path, device='cuda'):

    wave_s, sr_s = librosa.load(style_path, sr=24000)
    audio_s, index_s = librosa.effects.trim(wave_s, top_db=30)
    mel_tensor_s = preprocess(audio_s).to(device)

    wave_p, sr_p = librosa.load(style_path, sr=24000)
    audio_p, index_p = librosa.effects.trim(wave_p, top_db=30)
    mel_tensor_p = preprocess(audio_p).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor_s.unsqueeze(1)) # 1X128
        ref_p = model.predictor_encoder(mel_tensor_p.unsqueeze(1)) # 1X128

    ref_sp = torch.cat([ref_s, ref_p], dim=1) # 1X256

    return ref_sp

def compute_style_from_two_wavs(model, audio_s, audio_p, device='cuda'):

    mel_tensor_s = preprocess(audio_s).to(device)
    mel_tensor_p = preprocess(audio_p).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor_s.unsqueeze(1)) # 1X128
        ref_p = model.predictor_encoder(mel_tensor_p.unsqueeze(1)) # 1X128

    ref_sp = torch.cat([ref_s, ref_p], dim=1) # 1X256

    return ref_sp
