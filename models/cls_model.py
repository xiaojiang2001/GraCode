import torch
import torch.nn as nn
import clip

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes, clip_model='ViT-B/32', pretrained=True):
        super(CLIPClassifier, self).__init__()

        # Load CLIP model
        self.clip_model, _ = clip.load(clip_model, device="cuda")

        # Freeze CLIP model weights if pretrained
        if pretrained:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Get image encoder's output dimension
        image_encoding_dim = self.clip_model.visual.output_dim

        # Define fully connected layers for classification
        self.fc1 = nn.Linear(image_encoding_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        device = x.device  # Ensure tensors are on the same device
        # Use CLIP image encoder to extract features
        with torch.no_grad():  # Speed up if weights are frozen
            x = self.clip_model.encode_image(x).to(device)  # Ensure same dtype and device

        # Ensure dtype consistency for mixed precision
        x = x.float()  # Cast to float32 if needed

        # Pass through custom fully connected layers
        feature = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x, feature
