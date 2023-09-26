import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN_mini(nn.Module):
    '''
    This is a base CNN model.
    '''

    def __init__(self, feat_dim=256, pitch_class=13, pitch_octave=5):
        '''
        Definition of network structure.
        '''
        super().__init__()
        self.feat_dim = feat_dim
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # (8,64,250,64)  -> (8,250, 64,64) -> ( 8,250, 64*64) -
        # Position-wise feed-forward layer
        self.feed_forward = nn.Linear(64 * 64, 256)

        # assume x is now (8,250,256)

        # Prediction heads
        self.onset_head = nn.Linear(256, 1)  # needs to be (8,250,1) -> (8,250)
        self.offset_head = nn.Linear(256, 1)  # needs to be (8,250,1) -> (8,250)
        self.octave_head = nn.Linear(256, pitch_octave)  # needs to be (8,250,5)
        self.pitch_class_head = nn.Linear(256, pitch_class)  # needs to be (8,250,13)

    def forward(self, x):
        '''
        Compute output from input
        '''
        # Reshape x to have the expected shape (8, 1, 250, 256)
        x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = self.conv1(x) #shape becomes (8,16,250,256)
        x = self.bn1(x) #shape becomes (8,16,250,256)
        x = self.relu1(x) #shape becomes (8,16,250,256)
        x = self.pool1(x) #shape becomes (8,16,250,128)

        x = self.conv2(x) #shape becomes (8,32,250,128)
        x = self.bn2(x) #shape becomes (8,32,250,128)
        x = self.relu2(x) #shape becomes (8,32,250,128)
        x = self.pool2(x) #shape becomes (8,32,250,64)

        x = self.conv3(x) #shape becomes (8,64,250,64)
        x = self.bn3(x) #shape becomes (8,64,250,64)
        x = self.relu3(x) #shape becomes (8,64,250,64)

        # Position-wise feed-forward layer
        x = x.permute(0, 2, 1, 3)  #shape becomes (8,250,64,64)
        x = x.reshape(x.size(0), x.size(1), -1)  # Reshape to (8,250,64*64)
        x = self.feed_forward(x) # shape becomes (8,250,256)

        # Reshape logits to desired shapes
        onset_logits = self.onset_head(x) #shape becomes (8,250,1)
        onset_logits = onset_logits.reshape(onset_logits.size(0), -1) #shape becomes (8,250)

        offset_logits = self.offset_head(x) #shape becomes (8,250,1)
        offset_logits = offset_logits.reshape(onset_logits.size(0), -1) #shape becomes (8,250)

        octave_logits = self.octave_head(x) #shape becomes (8,250,5)
        pitch_class_logits = self.pitch_class_head(x) #shape becomes (8,250,13)

        return onset_logits, offset_logits, octave_logits, pitch_class_logits
