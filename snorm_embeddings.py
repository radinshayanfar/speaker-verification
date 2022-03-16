#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import random
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:

    # Train data (used for normalization)
    imposter_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["imposters_csv"], replacements={"data_root": data_folder},
    )

    # datasets = [train_data, enrol_data, test_data]
    datasets = [imposter_data]

    snt_len_sample = int(params["sample_rate"] * params["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("path", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(path, duration):
        duration_sample = int(duration * params["sample_rate"])
        if duration_sample > snt_len_sample:
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = 0
            stop = duration_sample
        num_frames = stop - start
        sig, fs = torchaudio.load(
            path, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        if sig.shape[0] < snt_len_sample:
            sig = F.pad(sig, (0, snt_len_sample - sig.shape[0]), "constant", 0)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # 4 Create dataloaders
    imposter_dataloader = sb.dataio.dataloader.make_dataloader(
        imposter_data, **params["imposter_dataloader_opts"]
    )

    # return train_dataloader, enrol_dataloader, test_dataloader
    return imposter_dataloader


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    imposter_dataloader = dataio_prep(params)
    # train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])
    params["mean_var_norm_emb"].eval()
    params["mean_var_norm_emb"].to(params["device"])

    # Computing  enrollment and test embeddings
    logger.info("computing imposter embeddings...")

    imposter_dict = compute_embedding_loop(imposter_dataloader)

    torch.save(torch.cat(list(imposter_dict.values()), dim=0),
        os.path.join(params["pretrain_path"], "imposter_embeddings.pt"))
    logger.info("saved imposter embeddings on pretrain_path")

    # # Compute the EER
    # logger.info("Computing EER..")
    # # Reading standard verification split
    # with open(veri_file_path) as f:
    #     veri_test = [line.rstrip() for line in f]

    # positive_scores, negative_scores = get_verification_scores(veri_test)
    # del enrol_dict, test_dict

    # eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    # logger.info("EER(%%)=%f", eer * 100)

    # min_dcf, th = minDCF(
    #     torch.tensor(positive_scores), torch.tensor(negative_scores)
    # )
    # logger.info("minDCF=%f", min_dcf * 100)

    # saving mean_var_norm_emb
    # logger.info("saving mean_var_norm_emb.ckpt to pretrain_path")
    # params["mean_var_norm_emb"]._save(os.path.join(params["pretrain_path"], "mean_var_norm_emb.ckpt"))
    # from speechbrain.utils.checkpoints import Checkpointer
    # checkpointer = Checkpointer('/content/tmp', {'mean_var_norm_emb': params["mean_var_norm_emb"]})
    # checkpointer.save_checkpoint()
    # params["mean_var_norm_emb"].glob_mean = torch.tensor([2.])
    # checkpointer.recover_if_possible()
    
    # from pprint import pprint
    # pprint(vars(params["mean_var_norm_emb"]))

    # import torch
    # import torchaudio

    # from speechbrain.pretrained import EncoderClassifier

    # verification = EncoderClassifier.from_hparams(source="/content/best_model/", hparams_file='hparams_inference.yaml')

    # signal1, sample_rate = torchaudio.load('/content/gdrive/MyDrive/SpeakerVerification/example/1.mp3')
    # emb = verification.encode_batch(signal1, normalize=False)
    # print(params["mean_var_norm_emb"].to('cpu')(
    #   emb, torch.ones(emb.shape[0], device=verification.device)
    # ))
