import torch
import torch.nn as nn
import torch.nn.functional as F

class CCLoss(nn.Module):
    def __init__(self, gamma, temperature=0.05, K=100, num_classes=45):
        super(CCLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.K = K
        self.num_classes = num_classes

    def forward(
        self,
        features,
        labels,
        sup_logits,
        gamma=0.05,
        temperature=0.05,
        queue_size=50,
        class_counts=None,
        logit_adjust_tau=0.0
    ):
        """
        Clean supervised contrastive classification loss.

        features:   [batch_size, feature_dim]
        labels:     [batch_size]
        sup_logits: [batch_size, num_classes]
        """

        device = features.device

        # ----------------------------------------------------
        # BASIC SANITY CHECKS
        # ----------------------------------------------------
        if labels.dim() > 1:
            labels = labels.view(-1)

        labels = labels.long().to(device)

        # ----------------------------------------------------
        # ALIGN CURRENT-BATCH FEATURES WITH LABELS AND LOGITS
        # ----------------------------------------------------
        #
        # The model may return additional queue/memory features.
        # Example:
        #   current batch: 20 features
        #   queue:         50 features
        #   total:         70 features
        #
        # Labels and supervised logits correspond only to the
        # current batch, so keep only that portion here.

        # sup_logits represents the current supervised batch.
        # features and labels may additionally contain queue/memory entries.
        batch_size = sup_logits.size(0)

        if labels.numel() < batch_size:
            raise RuntimeError(
                f"CCLoss size mismatch: labels batch={labels.numel()} "
                f"is smaller than logits batch={batch_size}"
            )

        if features.size(0) < batch_size:
            raise RuntimeError(
                f"CCLoss size mismatch: features batch={features.size(0)} "
                f"is smaller than logits batch={batch_size}"
            )

        # Keep only current-batch entries for supervised alignment.
        labels = labels[:batch_size]

        if features.size(0) > batch_size:
            features = features[:batch_size]

        num_classes = sup_logits.size(1)


        label_min = int(labels.min().detach().cpu())
        label_max = int(labels.max().detach().cpu())

        if label_min < 0 or label_max >= num_classes:
            raise RuntimeError(
                f"CCLoss label out of range: "
                f"min={label_min}, max={label_max}, "
                f"logit_classes={num_classes}"
            )

        # ----------------------------------------------------
        # CLASSIFICATION LOSS
        # ----------------------------------------------------
        if class_counts is not None and logit_adjust_tau > 0:
            counts = class_counts.to(device).float()

            if counts.numel() == num_classes:
                prior = counts / counts.sum().clamp_min(1.0)
                adjustment = logit_adjust_tau * torch.log(
                    prior.clamp_min(1e-12)
                )
                adjusted_logits = sup_logits + adjustment.unsqueeze(0)
            else:
                adjusted_logits = sup_logits
        else:
            adjusted_logits = sup_logits

        ce_loss = F.cross_entropy(adjusted_logits, labels)

        # ----------------------------------------------------
        # SUPERVISED CONTRASTIVE LOSS
        # ----------------------------------------------------

        # Normalize embeddings
        features = F.normalize(features, dim=1)

        # Pairwise similarity
        logits = torch.matmul(features, features.T) / temperature

        # Numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Remove self-comparisons
        self_mask = torch.eye(
            batch_size,
            device=device,
            dtype=torch.bool
        )

        logits_mask = ~self_mask

        # Positive pairs = same label, excluding self
        positive_mask = labels.unsqueeze(0).eq(
            labels.unsqueeze(1)
        ) & logits_mask

        # Denominator: all non-self examples
        exp_logits = torch.exp(logits) * logits_mask.float()

        denominator = exp_logits.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1e-12)

        log_prob = logits - torch.log(denominator)

        # Number of positive samples for each anchor
        positive_count = positive_mask.float().sum(dim=1)

        # Only anchors having at least one positive
        valid = positive_count > 0

        if valid.any():
            mean_log_prob_pos = (
                (positive_mask.float() * log_prob).sum(dim=1)
                / positive_count.clamp_min(1.0)
            )

            contrastive_loss = -mean_log_prob_pos[valid].mean()
        else:
            # No positive pairs in this batch
            contrastive_loss = torch.zeros(
                (),
                device=device,
                dtype=features.dtype
            )

        # ----------------------------------------------------
        # FINAL LOSS
        # ----------------------------------------------------
        loss = ce_loss + float(gamma) * contrastive_loss

        return loss


class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):

        ce = F.cross_entropy(
            logits,
            labels,
            reduction="none"
        )

        pt = torch.exp(-ce)

        loss = self.alpha * (1 - pt) ** self.gamma * ce

        return loss.mean()

def feature_decorrelation_loss(
    features,
    num_features=256,
    eps=1e-6
):
    """
    Lightweight feature decorrelation regularizer.
    """

    if features.dim() != 2:
        raise ValueError(
            f"Expected features with shape [B, D], got {features.shape}"
        )

    batch_size, feature_dim = features.shape

    if batch_size < 3:
        return features.new_zeros(())

    if feature_dim > num_features:
        indices = torch.randperm(
            feature_dim,
            device=features.device
        )[:num_features]

        z = features[:, indices]

    else:
        z = features

    # Center features
    z = z - z.mean(dim=0, keepdim=True)

    # Normalize feature dimensions
    std = z.std(
        dim=0,
        unbiased=False,
        keepdim=True
    )

    z = z / (std + eps)

    # Correlation matrix
    corr = torch.matmul(
        z.transpose(0, 1),
        z
    ) / batch_size

    # Remove diagonal
    diagonal = torch.diagonal(corr)

    off_diagonal = (
        corr - torch.diag_embed(diagonal)
    )

    denom = off_diagonal.numel() - z.size(1)

    if denom <= 0:
        return features.new_zeros(())

    return off_diagonal.pow(2).sum() / denom
