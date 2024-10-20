import argparse

import torch
from omegaconf import OmegaConf
from src.metrics import WER
from src.models import CTCModel
from tqdm import tqdm


def load_model(cfg, checkpoint_path):
    model = CTCModel(cfg)
    model.eval()
    model.freeze()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)

    return model


def calculate_wer(model, preds_list, encoded_len_list, targets_list):
    metric = WER()
    refs, hyps = [], []

    for pred_batch, encoded_len_batch, targets_batch in tqdm(
        zip(preds_list, encoded_len_list, targets_list), total=len(preds_list)
    ):
        for pred, pred_len, target in zip(
            pred_batch, encoded_len_batch, targets_batch
        ):
            hyps.append(
                model.decoder.decode_hypothesis(
                    pred[:pred_len], unique_consecutive=True
                )
            )
            refs.append(
                model.decoder.decode_hypothesis(
                    target, unique_consecutive=False
                )
            )

    metric.update(refs, hyps)
    wer = metric.compute()
    return wer[0]


def evaluate_model(model, dataloader):
    targets_list, target_len_list = [], []
    _, encoded_len_list, preds_list = [], [], []

    for i, batch in tqdm(enumerate(dataloader)):
        features, features_len, targets, target_len = batch

        with torch.inference_mode():
            _, encoded_len, preds = model.forward(
                features.to("cpu"), features_len.to("cpu")
            )

        targets_list.append(targets)
        target_len_list.append(target_len)

        encoded_len_list.append(encoded_len)

        preds_list.append(preds)
        if i == 5:
            break

    return calculate_wer(
        model, preds_list, encoded_len_list, targets_list
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate CTC Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("conf/ebranchformer_ctc.yaml")
    model = load_model(cfg, args.checkpoint)

    print("Evaluating on Opus dataset...")
    opus_wer = evaluate_model(model, model.val_dataloader())
    print("Opus WER:", opus_wer)

    cfg["val_dataloader"]["dataset"][
        "manifest_name"
    ] = "test_opus/farfield/manifest.jsonl"
    model = load_model(cfg, args.checkpoint)  

    print("Evaluating on farfield dataset...")
    farfield_wer = evaluate_model(model, model.val_dataloader())
    print("Farfield WER:", farfield_wer)


if __name__ == "__main__":
    main()
