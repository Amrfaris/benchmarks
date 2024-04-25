import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEEG(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(SimpleEEG, self).__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        T, C = input_shape[1], input_shape[2]  # Time points and channels
        print(input_shape)
        
        # Convolution layers
        self.temporal_conv = nn.Conv2d(1, 16, (1, T // 4), padding=(0, T // 8))
        self.spatial_conv = nn.Conv2d(16, 32, (C, 1), groups=16, padding=(C // 2, 0))
        
        # Pooling and dropout for regularization
        self.pooling = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the number of flattened features after convolutions and pooling
        self._to_linear = None
        self._dummy_input = torch.zeros(1, 1, C, T)  # Dummy input to calculate flat features
        self.convs = nn.Sequential(self.temporal_conv, self.spatial_conv, self.pooling, self.dropout)
        self._forward_convs(self._dummy_input)

        # Fully connected layer for classification
        self.fc1 = nn.Linear(self._to_linear, num_classes)

    def _forward_convs(self, x):
        x = self.convs(x)
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, x):
    # Ensuring input is in the format (batch_size, 1, channels, time_points)
      if x.shape[1] != 1:
          x = x.permute(0, 3, 2, 1)  # Correcting the shape if not matched
      #print(f"Corrected input shape before convolutions: {x.shape}")
      x = self._forward_convs(x)
      x = x.view(-1, self._to_linear)  # Flatten the features
      x = self.fc1(x)
      return F.log_softmax(x, dim=1)


