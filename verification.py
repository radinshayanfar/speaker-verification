import os
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition


class SpeakerVerification(SpeakerRecognition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    @classmethod
    def from_hparams(cls, *args, **kwargs):
        verification = super(cls,cls).from_hparams(*args, **kwargs)
        source = kwargs['source']
        if os.path.exists(os.path.join(source, 'imposter_embeddings.pt')):
            verification.imp_emb = torch.load(os.path.join(source, 'imposter_embeddings.pt'), map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        return verification
    
    def compute_snorm(self, emb1, emb2):
        emb1 = emb1.squeeze(0)
        emb2 = emb2.squeeze(0)
        score_e1 = self.similarity(emb1, self.imp_emb)
        score_e2 = self.similarity(emb2, self.imp_emb)
        score_e1_e2 = self.similarity(emb1, emb2)
        score_e1_normed = (score_e1_e2 - score_e1.mean()) / score_e1.std()
        score_e2_normed = (score_e1_e2 - score_e2.mean()) / score_e2.std()
        return score_e1_normed + score_e2_normed
    
    @staticmethod
    def __segment_to_tensor(segment):
        segment = segment.set_frame_rate(16000)
        tensor = torch.Tensor(segment.get_array_of_samples())
        tensor = tensor.unsqueeze(dim=0)
        return tensor

    def embed_segment(self, segment, mean_norm=True, a_norm=True):
        batch = SpeakerVerification.__segment_to_tensor(segment)
        # Amplitude Norm
        if a_norm:
            batch = self.rms_normalize(batch)
        # Embed
        emb = self.encode_batch(batch, normalize=mean_norm)
        return emb
    
    def score_embeddings(self, emb1, emb2, threshold=10, snorm=True):
        # SNorm
        if snorm and hasattr(self, 'imp_emb'):
            score = self.compute_snorm(emb1, emb2)
        else:
            score = self.similarity(emb1, emb2)
        # Decision
        decision = score > threshold
        # Squeeze
        return score[0], decision[0]
    
    def peak_normalize(self, sig):
        return sig / sig.abs().max()
      
    def rms_normalize(self, sig, rms_level=0):
        """
        Normalize the signal with rms technique.
        Args:
            - sig       (torch.Tensor) : input signal
            - rms_level (int) : rms level in dB.
        """
        # linear rms level and scaling factor
        r = 10**(rms_level / 10.0)
        a = torch.sqrt( (len(sig) * r**2) / torch.sum(sig**2) )

        # normalize
        return sig * a
          
    def verify_files(self, path_x, path_y, threshold=10, mean_norm=True, snorm=True, a_norm=True):
        """Speaker verification with cosine distance
        Returns the score and the decision (0 different speakers,
        1 same speakers).
        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        batch_x, _ = torchaudio.load(path_x)
        batch_y, _ = torchaudio.load(path_y)

        if a_norm:
            batch_x = self.rms_normalize(batch_x)
            batch_y = self.rms_normalize(batch_y)
        # Verify:
        emb1 = self.encode_batch(batch_x, normalize=mean_norm)
        emb2 = self.encode_batch(batch_y, normalize=mean_norm)
        # SNorm
        if snorm and hasattr(self, 'imp_emb'):
            score = self.compute_snorm(emb1, emb2)
        else:
            score = self.similarity(emb1, emb2)
        decision = score > threshold
        # Squeeze:
        return score[0], decision[0]
