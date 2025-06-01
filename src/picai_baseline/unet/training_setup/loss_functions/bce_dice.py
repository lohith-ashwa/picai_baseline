import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        """
        Combined BCE and Dice Loss.
        Args:
            bce_weight (float): Weight for BCE. (1 - bce_weight) is used for Dice.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()  # Includes sigmoid inside

    def forward(self, inputs, targets):
        # Compute BCE Loss (includes sigmoid inside)
        bce_loss = self.bce(inputs, targets.float())

        # Apply sigmoid to inputs for Dice calculation
        probs = torch.sigmoid(inputs)

        # Flatten for Dice
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # Combine
        total_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return total_loss
