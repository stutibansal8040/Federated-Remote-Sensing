import math
import torch


def get_class_counts(dataset, num_classes):
    """
    Calculate the number of samples belonging to each class
    in a client's local dataset.

    Supports common PyTorch dataset formats:
      - dataset.targets
      - dataset.labels
      - dataset.samples = [(path, label), ...]
    """

    class_counts = [0] * num_classes

    # Case 1: torchvision-style dataset
    if hasattr(dataset, "targets"):
        targets = dataset.targets

        if torch.is_tensor(targets):
            targets = targets.tolist()

        for label in targets:
            label = int(label)
            if 0 <= label < num_classes:
                class_counts[label] += 1

        return class_counts

    # Case 2: dataset has labels
    if hasattr(dataset, "labels"):
        labels = dataset.labels

        if torch.is_tensor(labels):
            labels = labels.tolist()

        for label in labels:
            label = int(label)
            if 0 <= label < num_classes:
                class_counts[label] += 1

        return class_counts

    # Case 3: torchvision ImageFolder-style dataset
    if hasattr(dataset, "samples"):
        for _, label in dataset.samples:
            label = int(label)
            if 0 <= label < num_classes:
                class_counts[label] += 1

        return class_counts

    raise ValueError(
        "Could not find labels in dataset. "
        "Expected dataset.targets, dataset.labels, or dataset.samples."
    )


def calculate_normalized_entropy(class_counts):
    """
    Calculate normalized Shannon entropy.

    H = -sum(p_c log(p_c)) / log(C)

    Range:
        0 = completely skewed
        1 = perfectly balanced
    """

    total = sum(class_counts)

    if total == 0:
        return 0.0

    probabilities = [
        count / total
        for count in class_counts
        if count > 0
    ]

    entropy = 0.0

    for p in probabilities:
        entropy -= p * math.log(p)

    num_classes = len(class_counts)

    if num_classes <= 1:
        return 0.0

    max_entropy = math.log(num_classes)

    normalized_entropy = entropy / max_entropy

    # Numerical safety
    normalized_entropy = max(
        0.0,
        min(1.0, normalized_entropy)
    )

    return normalized_entropy


def calculate_adaptive_gamma(
    entropy,
    gamma_min=0.02,
    gamma_max=0.10
):
    """
    Convert client entropy into adaptive gamma.

    Low entropy  -> high gamma
    High entropy -> low gamma
    """

    gamma = (
        gamma_min
        + (gamma_max - gamma_min)
        * (1.0 - entropy)
    )

    return gamma


def get_client_adaptive_gamma(
    dataset,
    num_classes,
    gamma_min=0.02,
    gamma_max=0.10
):
    """
    Complete pipeline:

        dataset
           ↓
        class counts
           ↓
        entropy
           ↓
        adaptive gamma
    """

    class_counts = get_class_counts(
        dataset,
        num_classes
    )

    entropy = calculate_normalized_entropy(
        class_counts
    )

    gamma = calculate_adaptive_gamma(
        entropy,
        gamma_min,
        gamma_max
    )

    return class_counts, entropy, gamma
