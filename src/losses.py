import torch
import torch.nn as nn
import torch.nn.functional as F

class CCLoss(nn.Module):
    def __init__(self, gamma, temperature=0.05, K=100, num_classes=21):
        super(CCLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.K = K
        self.num_classes = num_classes

    def forward(
        self,
        features,
        labels=None,
        sup_logits=None,
        gamma=None,
        temperature=None,
        queue_size=None,
        class_counts=None,
        logit_adjust_tau=0.0
    ):
        device = features.device

        if gamma is None:
            gamma = self.gamma

        if temperature is None:
            temperature = self.temperature

        if queue_size is None:
            queue_size = self.K

        queue_size = int(queue_size)

        if queue_size <= 0:
            raise ValueError(
                f"queue_size must be positive, got {queue_size}"
            )

        if queue_size > self.K:
            raise ValueError(
                f"queue_size={queue_size} exceeds "
                f"maximum K={self.K}"
            )

        # features contains:
        #   Q features
        #   K positive features
        #   physical queue of size self.K
        #
        # Only the first queue_size entries of the
        # physical queue participate in this round.
        total_feature_dim = features.shape[0]

        # Feature layout from MoCo:
        # [B query features] + [B key features] + [K queue features]
        #
        # Therefore:
        # total_feature_dim = 2 * bs + self.K
        remaining_features = total_feature_dim - self.K

        if remaining_features <= 0:
            raise ValueError(
                f"Invalid feature layout: "
                f"features={features.shape}, K={self.K}"
            )

        if remaining_features % 2 != 0:
            raise ValueError(
                f"Invalid feature layout: expected 2*bs + K entries, "
                f"got features={total_feature_dim}, K={self.K}"
            )

        bs = remaining_features // 2

        if bs <= 0:
            raise ValueError(
                f"Invalid feature layout: "
                f"features={features.shape}, K={self.K}"
            )

        queue_start = 2 * bs
        queue_end = queue_start + queue_size

        if queue_end > total_feature_dim:
            raise ValueError(
                f"queue_end={queue_end} exceeds "
                f"feature dimension={total_feature_dim}"
            )

        # Keep Q + K positives and only the active queue.
        active_features = torch.cat(
            (
                features[:2 * bs],
                features[queue_start:queue_end]
            ),
            dim=0
        )

        feature_sim = (
            torch.matmul(
                features[:bs],
                active_features.T
            ) / temperature
        )

        # The contrastive similarity is computed only for the B query
        # features. The supervised logits may contain entries for both
        # query and key features (2B), so keep only the query portion.
        if sup_logits is None:
            raise ValueError("sup_logits must not be None")

        if sup_logits.shape[0] == 2 * bs:
            sup_logits = sup_logits[:bs]
        elif sup_logits.shape[0] != bs:
            raise ValueError(
                f"Invalid sup_logits shape: {sup_logits.shape}; "
                f"expected first dimension {bs} or {2 * bs}"
            )

        # -------------------------------------------------
        # Local class-frequency logit adjustment
        # -------------------------------------------------
        # p_c = n_c / sum(n_c)
        # adjusted_logit_c = logit_c - tau * log(p_c)
        #
        # This compensates for the bias toward classes that
        # occur more frequently on an individual client.
        if class_counts is not None and logit_adjust_tau > 0:
            # class_counts may come from the client as a
            # Python list, NumPy array, or PyTorch tensor.
            class_counts = torch.as_tensor(
                class_counts,
                device=sup_logits.device,
                dtype=sup_logits.dtype
            )

            class_probs = class_counts / class_counts.sum().clamp_min(1.0)
            class_probs = class_probs.clamp_min(1e-12)

            log_prior = torch.log(class_probs)

            if log_prior.numel() != sup_logits.shape[1]:
                raise ValueError(
                    f"class_counts has {log_prior.numel()} classes, "
                    f"but sup_logits has {sup_logits.shape[1]} classes"
                )

            sup_logits = sup_logits - logit_adjust_tau * log_prior.unsqueeze(0)

        logits_con = torch.cat(
            (sup_logits, feature_sim),
            dim=1
        )
        logits_max, _ = torch.max(logits_con, dim=1, keepdim=True)
        logits = logits_con - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)

        active_labels = torch.cat(
            (
                labels[:2 * bs],
                labels[2 * bs:2 * bs + queue_size]
            ),
            dim=0
        )

        con_mask = torch.eq(
            labels[:bs],
            active_labels.T
        ).float().to(device)

        logits_mask = torch.ones_like(con_mask)

        # Remove the query's own positive position.
        logits_mask[:, 0] = 0

        e_mask = con_mask * logits_mask

        one_hot_label = torch.nn.functional.one_hot(
            labels[:bs].view(-1,),
            num_classes=self.num_classes
        ).to(torch.float32)

        df_mask = torch.cat(
            (
                one_hot_label,
                e_mask * gamma
            ),
            dim=1
        )

        logits_mask = torch.cat(
            (
                torch.ones(
                    bs,
                    self.num_classes,
                    device=device
                ),
                logits_mask
            ),
            dim=1
        )
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (df_mask * log_prob).sum(1) / df_mask.sum(1)
        
        loss = - mean_log_prob_pos.mean()
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