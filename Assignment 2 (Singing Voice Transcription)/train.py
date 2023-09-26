import torch
import torch.nn as nn
import torch.optim as optim
from torch import float32
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_squared_error

from model import BaseCNN_mini
from dataset import get_data_loader, move_data_to_device
from utils import ls
# import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


# os.environ["CUDA_VISIBLE_DEVICES"] = '3' # If you have multiple GPU's,
# uncomment this line to specify which GPU you want to use


def main(epoch=10):
    # converting to windows paths
    args = {
        # 'save_model_dir': './results',
        'save_model_dir': '.\\results',
        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'device': 'cpu',
        'dataset_root': '.\\data_mini\\',
        'sampling_rate': 16000,
        'sample_length': 5,  # in second
        # 'num_workers': 4,  # ironically making this 0 speeds up my dataloader!
        'num_workers': 0,  # Number of additional thread for data loading. A large number may freeze your laptop.
        'annotation_path': '.\\data_mini\\annotations.json',
        'frame_size': 0.02,
        'batch_size': 8,  # 32 produce best result so far
        'best_model_path': '/best_model.pth' if epoch == 10 else f'/best_model_{epoch}.pth'
    }

    ast_model = AST_Model(args['device'])

    # Set learning params
    learning_params = {
        'batch_size': 50,
        'epoch': epoch,
        'lr': 1e-4,
    }

    # Train and Validation
    best_model_id = ast_model.fit(args, learning_params)
    print("Best model from epoch: ", best_model_id)


class AST_Model:
    '''
    This is main class for training model and making predictions.
    '''

    def __init__(self, device="cpu", model_path=None):
        # Initialize model
        self.device = device
        self.model = BaseCNN_mini(feat_dim=256).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded.')
        else:
            print('Model initialized.')

    def fit(self, args, learning_params):
        # Set paths
        save_model_dir = args['save_model_dir']
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_params['lr'])
        loss_func = LossFunc(device=self.device)
        metric = Metrics(loss_func)

        train_loader = get_data_loader(split='train', args=args)
        valid_loader = get_data_loader(split='valid', args=args)

        # Start training
        print('Start training...')
        start_time = time.time()
        best_model_id = -1
        min_valid_loss = 10000

        for epoch in range(1, learning_params['epoch'] + 1):
            self.model.train()
            total_training_loss = 0

            # Train
            pbar = tqdm(train_loader)
            for batch_idx, batch in enumerate(pbar):
                x, onset, offset, octave, pitch_class = move_data_to_device(batch, args['device'])
                tgt = onset, offset, octave, pitch_class
                out = self.model(x)
                losses = loss_func.get_loss(out, tgt)
                loss = losses[0]
                metric.update(out, tgt, losses)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()

                pbar.set_description('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
            metric_train = metric.get_value()

            # Validation
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_loader):
                    x, onset, offset, octave, pitch_class = move_data_to_device(batch, args['device'])
                    tgt = onset, offset, octave, pitch_class
                    out = self.model(x)
                    metric.update(out, tgt)
            metric_valid = metric.get_value()

            # Logging
            print('[Epoch {:02d}], Train Loss: {:.5f}, Valid Loss {:.5f}, Time {:.2f}s'.format(
                epoch, metric_train['loss'], metric_valid['loss'], time.time() - start_time,
            ))
            print('Split Train F1/Accuracy: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_train['onset_f1'],
                metric_train['offset_f1'],
                metric_train['octave_acc'],
                metric_train['pitch_acc']
            ))
            print('Split Valid F1/Accuracy: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_valid['onset_f1'],
                metric_valid['offset_f1'],
                metric_valid['octave_acc'],
                metric_valid['pitch_acc']
            ))
            print('Split Train Loss: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_train['onset_loss'],
                metric_train['offset_loss'],
                metric_train['octave_loss'],
                metric_train['pitch_loss']
            ))
            print('Split Valid Loss: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_valid['onset_loss'],
                metric_valid['offset_loss'],
                metric_valid['octave_loss'],
                metric_valid['pitch_loss']
            ))

            # Save the best model
            if metric_valid['loss'] < min_valid_loss:
                min_valid_loss = metric_valid['loss']
                best_model_id = epoch

                save_dict = self.model.state_dict()
                best_model_path = args["best_model_path"]
                # target_model_path = save_model_dir + '/best_model.pth'
                target_model_path = save_model_dir + best_model_path
                torch.save(save_dict, target_model_path)

        print('Training done in {:.1f} minutes.'.format((time.time() - start_time) / 60))
        return best_model_id

    def parse_frame_info(self, frame_info, args):
        """
        Convert frame-level output into note-level predictions.
        """

        frame_num = len(frame_info)

        result = []
        current_onset = None
        pitch_counter = []
        local_max_size = 3
        current_frame = 0.0

        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
        onset_seq_length = len(onset_seq)

        frame_length = args['frame_size']

        for i in range(frame_num):  # For each frame
            current_frame = frame_length * i
            info = frame_info[i]
            last_frame = max(0, current_frame - 1)

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # If the frame is an onset
            if info[0]:
                if current_onset is None:
                    current_onset = current_frame
                else:
                    if len(pitch_counter) > 0:
                        pitch = max(set(pitch_counter), key=pitch_counter.count) + 36
                        result.append([current_onset, current_frame, pitch])
                    current_onset = current_frame
                    pitch_counter = []

            # If it is offset
            elif info[1]:
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        pitch = max(set(pitch_counter), key=pitch_counter.count) + 36
                        result.append([current_onset, current_frame, pitch])
                    current_onset = None
                    pitch_counter = []
                else:
                    pass

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                if info[2] != 0 and info[3] != 0:
                    current_pitch = int((info[2] - 1) * 12 + (info[3] - 1))
                    pitch_counter.append(current_pitch)

        # The last note
        if current_onset is not None:
            if len(pitch_counter) > 0:
                pitch = max(set(pitch_counter), key=pitch_counter.count) + 36
                result.append([current_onset, current_frame, pitch])

        return result

    def predict(self, testset_path, onset_thres, offset_thres, args):
        """Predict results for a given test dataset."""
        songs = ls(testset_path)
        results = {}
        for song in songs:
            if song.startswith('.'):
                continue
            test_loader = get_data_loader(split='test', fns=[song], args=args)

            # Start predicting
            self.model.eval()
            with torch.no_grad():
                on_frame = []
                off_frame = []
                oct_frame = []
                pitch_frame = []
                loss_func = LossFunc(args['device'])
                metric = Metrics(loss_func)
                pbar = tqdm(test_loader)
                for batch_idx, batch in enumerate(pbar):
                    x, onset, offset, octave, pitch_class = move_data_to_device(batch, self.device)
                    tgt = onset, offset, octave, pitch_class
                    out = self.model(x)
                    metric.update(out, tgt)

                    # Collect frames for corresponding songs
                    on_out = out[0].flatten()
                    on_out[on_out >= onset_thres] = 1
                    on_out[on_out < onset_thres] = 0
                    on_out = on_out.long()
                    off_out = out[1].flatten()
                    off_out[off_out >= offset_thres] = 1
                    off_out[off_out < offset_thres] = 0
                    off_out = off_out.long()
                    oct_out = torch.argmax(out[2], dim=2).flatten()
                    pitch_out = torch.argmax(out[3], dim=2).flatten()

                    on_frame.append(on_out)
                    off_frame.append(off_out)
                    oct_frame.append(oct_out)
                    pitch_frame.append(pitch_out)

                on_out = torch.cat(on_frame).tolist()
                off_out = torch.cat(off_frame).tolist()
                oct_out = torch.cat(oct_frame).tolist()
                pitch_out = torch.cat(pitch_frame).tolist()
                frame_info = list(zip(on_out, off_out, oct_out, pitch_out))

                # Parse frame info into output format for every song
                results[song] = self.parse_frame_info(frame_info=frame_info, args=args)

        return results


class LossFunc:
    def __init__(self, device):
        self.device = device
        '''
        We will use Binary Cross Entropy for onset and offset classification, with 15 as weight for positive value
        Cross Entropy Loss for octave and pitch class classification.
        
        NOTE: raw model output will not be normalized by softmax/sigmoid function. 
        
        YOUR CODE: finish the __init__ and get_loss function.
        '''
        self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], dtype=torch.float32))
        self.offset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], dtype=torch.float32))
        self.octave_criterion = nn.CrossEntropyLoss()
        self.pitch_criterion = nn.CrossEntropyLoss()
        self.num_classes_octave = 5
        self.num_classes_pitch = 13

    def one_hotify(self, tensor, num_classes):
        # Preprocess to make it a one hot vector
        new_tensor = torch.zeros(*tensor.shape, num_classes)

        # Convert each y value to a one-hot vector
        for i in range(num_classes):
            new_tensor[:, :, i] = (tensor == i).float()

        return new_tensor

    def get_loss(self, out, tgt):
        '''
        This function receive model output and target for onset, offset, octave, and pitch class, then
        compute loss for the 4 pairs respectively,
        finally add them together (simple addition, no weight) as the total loss
        Return: (total loss, onset loss, offset loss, octave loss, pitch loss)
        '''

        '''
        YOUR CODE HERE: finish the function
        '''
        on_out, off_out, octave_out, pitch_class_out = out
        on_tgt, off_tgt, octave_tgt, pitch_class_tgt = tgt

        # ensuring the dtypes are all the same!
        on_out = on_out.to(torch.float32)
        on_tgt = on_tgt.to(torch.float32)

        off_out = off_out.to(torch.float32)
        off_tgt = off_tgt.to(torch.float32)

        octave_out = octave_out.to(torch.float32)
        octave_tgt = octave_tgt.to(torch.float32)
        # Preprocess to make it a one hot vector
        octave_tgt = self.one_hotify(octave_tgt, self.num_classes_octave)
        # flatten it to use in Cross Entropy
        octave_out = octave_out.view(-1, octave_out.size(2))
        octave_tgt = octave_tgt.view(-1, octave_tgt.size(2))

        pitch_class_out = pitch_class_out.to(torch.float32)
        pitch_class_tgt = pitch_class_tgt.to(torch.float32)
        # Preprocess to make it a one hot vector
        pitch_class_tgt = self.one_hotify(pitch_class_tgt, self.num_classes_pitch)
        # flatten it to use in Cross Entropy
        pitch_class_out = pitch_class_out.view(-1, pitch_class_out.size(2))
        pitch_class_tgt = pitch_class_tgt.view(-1, pitch_class_tgt.size(2))

        # Now calculate individual loss
        onset_loss = self.onset_criterion(on_out, on_tgt)
        offset_loss = self.onset_criterion(off_out, off_tgt)
        octave_loss = self.octave_criterion(octave_out, octave_tgt)
        pitch_loss = self.pitch_criterion(pitch_class_out, pitch_class_tgt)

        # computing total loss!
        total_loss = onset_loss + offset_loss + octave_loss + pitch_loss

        return (total_loss, onset_loss, offset_loss, octave_loss, pitch_loss)


class Metrics:
    def __init__(self, loss_func):
        self.buffer = {}
        self.loss_func = loss_func
        self.num_classes_octave = 5
        self.num_classes_pitch = 13
        self.threshold = 0.5

    def one_hotify(self, tensor, num_classes):
        # Preprocess to make it a one hot vector
        new_tensor = torch.zeros(*tensor.shape, num_classes, dtype=float32)

        # Convert each y value to a one-hot vector
        for i in range(num_classes):
            new_tensor[:, :, i] = (tensor == i).float()

        return new_tensor

    def update(self, out, tgt, losses=None):
        '''
        Compute metrics for one batch of output and target.
        F1 score for onset and offset,
        Accuracy for octave and pitch class.
        Append the results to a list,
        and link the list to self.buffer[metric_name].
        '''
        with torch.no_grad():
            out_on, out_off, out_oct, out_pitch = out
            tgt_on, tgt_off, tgt_oct, tgt_pitch = tgt

            if losses == None:
                losses = self.loss_func.get_loss(out, tgt)

            # ensuring the dtypes are all the same!
            out_on = torch.tensor(out_on, dtype=float32)
            tgt_on = torch.tensor(tgt_on, dtype=float32)

            out_off = torch.tensor(out_off, dtype=float32)
            tgt_off = torch.tensor(tgt_off, dtype=float32)

            out_oct = torch.tensor(out_oct, dtype=float32)
            tgt_oct = torch.tensor(tgt_oct, dtype=float32)

            out_pitch = torch.tensor(out_pitch, dtype=float32)
            tgt_pitch = torch.tensor(tgt_pitch, dtype=float32)

            # Preprocess to make it a one hot vector
            tgt_oct = self.one_hotify(tgt_oct, self.num_classes_octave)
            tgt_pitch = self.one_hotify(tgt_pitch, self.num_classes_pitch)

            # converting onset and offset values to sigmoid to get it in the range of 0-> 1
            out_on = torch.sigmoid(out_on)
            tgt_on = torch.sigmoid(tgt_on)
            out_off = torch.sigmoid(out_off)
            tgt_off = torch.sigmoid(tgt_off)

            # converting pitch and octave to softmax to get it like a proper one hot encoding!
            out_oct = F.softmax(out_oct, dim=2)
            tgt_oct = F.softmax(tgt_oct, dim=2)

            out_pitch = F.softmax(out_pitch, dim=2)
            tgt_pitch = F.softmax(tgt_pitch, dim=2)

            '''
            YOUR CODE HERE: compute the four metrics below.
            '''
            # Compute accuracy for octave
            predicted_labels_oct = torch.argmax(out_oct, dim=2)
            target_labels_oct = torch.argmax(tgt_oct, dim=2)
            correct_predictions_oct = torch.sum(predicted_labels_oct == target_labels_oct)
            total_predictions_oct = out_oct.size(0) * out_oct.size(1)  # Total number of predictions
            oct_acc = correct_predictions_oct.item() / total_predictions_oct

            # Compute accuracy for pitch
            predicted_labels_pitch = torch.argmax(out_pitch, dim=2)
            target_labels_pitch = torch.argmax(tgt_pitch, dim=2)
            correct_predictions_pitch = torch.sum(predicted_labels_pitch == target_labels_pitch)
            total_predictions_pitch = out_pitch.size(0) * out_pitch.size(1)
            pitch_acc = correct_predictions_pitch.item() / total_predictions_pitch

            # For onset and offset compute f1 score, just flatten the vectors for the whole batch of 8 , threshold it and compute the f1 score directly for a set of 8 batches!
            out_on = (out_on > self.threshold).to(torch.int32)
            out_on = out_on.view(-1)
            tgt_on = (tgt_on > self.threshold).to(torch.int32)
            tgt_on = tgt_on.view(-1)
            onset_f1 = f1_score(out_on.cpu().numpy(), tgt_on.cpu().numpy())

            out_off = (out_off > self.threshold).to(torch.int32)
            out_off = out_off.view(-1)
            tgt_off = (tgt_off > self.threshold).to(torch.int32)
            tgt_off = tgt_off.view(-1)
            offset_f1 = f1_score(out_off.cpu().numpy(), tgt_off.cpu().numpy())

            batch_metric = {
                'loss': losses[0].item(),
                'onset_loss': losses[1].item(),
                'offset_loss': losses[2].item(),
                'octave_loss': losses[3].item(),
                'pitch_loss': losses[4].item(),
                'onset_f1': onset_f1,
                'offset_f1': offset_f1,
                'octave_acc': oct_acc,
                'pitch_acc': pitch_acc,
            }

            for k in batch_metric:
                if k in self.buffer:
                    self.buffer[k].append(batch_metric[k])
                else:
                    self.buffer[k] = [batch_metric[k]]

    def get_value(self):
        for k in self.buffer:
            self.buffer[k] = sum(self.buffer[k]) / len(self.buffer[k])
        ret = self.buffer
        self.buffer = {}
        return ret


if __name__ == '__main__':
    main(epoch=30)
