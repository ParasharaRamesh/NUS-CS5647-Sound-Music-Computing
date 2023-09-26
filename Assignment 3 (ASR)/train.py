#!/usr/bin/env/python3
"""This minimal example trains a CTC-based speech recognizer on a tiny dataset.
The encoder is based on a combination of convolutional, recurrent, and
feed-forward networks (CRDNN) that predict phonemes.  A greedy search is used on
top of the output probabilities.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).
"""
import sys
import pathlib
import random

import speechbrain
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from utils import *
import torch
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.nnet.schedulers import NewBobScheduler


def main(device="cpu"):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparam_fn = sys.argv[1]
    hparams_file = experiment_dir / hparam_fn
    data_folder = './datasets/tiny_librispeech'
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    # NOTE (Task 3.1) test data is also taken into account here
    train_data, valid_data, test_data = data_prep(data_folder, hparams)

    # Trainer initialization
    # NOTE (Task 3.2) checkpointer is defined here
    os.makedirs(hparams["checkpoint_dir"], exist_ok=True)
    checkpointer = Checkpointer(hparams["checkpoint_dir"])
    # NOTE add another param in this dict to keep track
    hparams["best_PER_after_train"] = float("inf")

    ctc_brain = CTCBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        checkpointer=checkpointer,
        run_opts={"device": device}
    )

    # Training/validation loop
    ctc_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # NOTE. (Task 3.3) Load the best training PER model checkpoint(most recent) for evaluation
    bestCheckpoint = ctc_brain.checkpointer.find_checkpoint()
    ctc_brain.checkpointer.load_checkpoint(bestCheckpoint)

    # Evaluation is run separately (now just evaluating on valid data)
    # NOTE. (Task 3.1) using test_data for evaluation instead of valid_data which was used before
    ctc_brain.evaluate(test_data, min_key='PER')


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    annot_dir = jpath(data_folder, 'annotation')
    tokenizer = PhonemeTokenizer()

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / jpath(annot_dir, "train.json"),
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / jpath(annot_dir, "valid.json"),
        replacements={"data_root": data_folder},
    )
    # NOTE:(Task 3.1) Added test dataset
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / jpath(annot_dir, "test.json"),
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        phn_list = [a.upper() for a in phn_list]
        yield phn_list
        phn_encoded = tokenizer.encode_seq(phn_list)
        phn_encoded = torch.tensor(phn_encoded, dtype=torch.long)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

    # NOTE. (Task 3.1) returning the test data as well
    return train_data, valid_data, test_data


class CTCBrain(sb.Brain):
    def on_fit_start(self):
        super().on_fit_start()  # resume ckpt
        self.tokenizer = PhonemeTokenizer()

        # NOTE: (Task 4.8.1) initialize the scheduler learning rate
        self.scheduler = NewBobScheduler(initial_value=self.optimizer.defaults["lr"])

        # NOTE: (Task 4.8.1) store the current validation loss for this epoch and keep changing this!
        self.curent_validation_loss = float("inf")

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        predictions, lens = predictions
        phns, phn_lens = batch.phn_encoded
        loss = self.hparams.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            seq = sb.decoders.ctc_greedy_decode(
                predictions, lens, blank_id=self.hparams.blank_index
            )
            t = phns.tolist()
            out = self.tokenizer.decode_seq_batch(seq)
            tgt = self.tokenizer.decode_seq_batch(t)
            self.per_metrics.append(batch.id, out, tgt)

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        elif stage == sb.Stage.VALID:
            valid_PER = self.per_metrics.summarize("error_rate")
            summary = self.per_metrics.summary
            optimizer_current_state = self.optimizer

            # NOTE (Task 4.8.2) print current lr
            current_lr = optimizer_current_state.param_groups[0]["lr"]
            stats = f"|Stage: {stage}|Epoch #: {epoch}|Train loss: {self.train_loss}|Loss: {stage_loss}|PER: {valid_PER}|Current LR: {current_lr}|\n"
            print("+" * 300)
            print(stats)
            print(summary)
            print("-" * 300)

            # NOTE (Task 4.8.1) compare the current val loss with the stage loss, if stage loss is greater ( i.e. val loss didnt decrease) use the scheduler to change lr else ignore
            if (self.curent_validation_loss != None) and (stage_loss > self.curent_validation_loss):
                # some random number
                _, new_lr = self.scheduler(metric_value=random.randint(1, 10))
                print(f"Using the scheduler to change learning rate from {current_lr} -> {new_lr} ")

                # changes the learning rate to something else!
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # NOTE (Task 4.8.1) overwrite the current epoch's validation loss
            self.curent_validation_loss = stage_loss

            # NOTE (Task 3.5) need to log stats ( writing in append mode)
            with open(self.hparams.train_valid_result_fn, "a") as train_valid_file:
                # write stats
                train_valid_file.write(stats)

                # write summary
                train_valid_file.write(str(summary) + "\n")

                # write line
                train_valid_file.write("-" * 400 + "\n")

            # NOTE (Task 3.2) Check if the current training PER is lower than the previous best training PER
            if valid_PER < self.hparams.best_PER_after_train:
                # Save the best model checkpoint during training
                self.hparams.best_PER_after_train = valid_PER
                self.checkpointer.save_checkpoint()

        elif stage == sb.Stage.TEST:
            test_PER = self.per_metrics.summarize("error_rate")
            test_summary = self.per_metrics.summary
            test_stats = f"|Stage: {stage}|Loss: {self.train_loss}|PER: {test_PER}|\n"
            print("x" * 300)
            print(test_stats)
            print(test_summary)

            # NOTE (Task 3.4) Store the output of the model in this file
            with open(self.hparams.result_fn, "w") as result:
                self.per_metrics.write_stats(result)
                result.write("-" * 200 + "\n\n")
                result.write("Summary: \n" + str(test_summary) + "\n")


if __name__ == "__main__":
    main()
