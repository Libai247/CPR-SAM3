"""Unchanged prototype sampling, pooling and clustering helpers."""

from pathlib import Path
import torch
import torch.nn.functional as F


def image_paths(directory, maximum):
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if maximum > 0 and len(paths) > maximum:
        indices = torch.linspace(0, len(paths) - 1, maximum).round().long()
        paths = [paths[index] for index in indices.tolist()]
    return paths


def image_prototypes(logits, features, seed_tokens):
    low = F.interpolate(
        logits.unsqueeze(0),
        size=features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    top2 = low.topk(min(2, low.shape[0]), dim=0)
    labels = top2.indices[0]
    if low.shape[0] > 1:
        margin = (
            (top2.values[0] - top2.values[1])
            / top2.values[0].abs().clamp_min(0.05)
        )
    else:
        margin = torch.ones_like(top2.values[0])
    one_hot = F.one_hot(labels, num_classes=low.shape[0]).permute(
        2, 0, 1
    ).float()
    coherence = F.avg_pool2d(
        one_hot.unsqueeze(0), 3, stride=1, padding=1
    ).squeeze(0).gather(0, labels.unsqueeze(0)).squeeze(0)
    quality = margin.clamp(0.0, 2.0) * coherence
    tokens = features.flatten(1).transpose(0, 1).float()
    flat_labels = labels.flatten()
    flat_quality = quality.flatten()
    records = []
    for class_index in range(low.shape[0]):
        indices = torch.nonzero(
            flat_labels.eq(class_index) & flat_quality.gt(0.20),
            as_tuple=False,
        ).flatten()
        if indices.numel() < 8:
            continue
        count = min(int(seed_tokens), indices.numel())
        selected = indices[flat_quality[indices].topk(count).indices]
        weights = torch.softmax(flat_quality[selected] / 0.10, dim=0)
        prototype = F.normalize(
            (tokens[selected] * weights.unsqueeze(1)).sum(dim=0),
            dim=0,
            eps=1e-6,
        )
        records.append(
            (
                class_index,
                prototype.cpu(),
                float(flat_quality[selected].mean().item()),
            )
        )
    return records


def clustered_prototypes(records, classes, clusters, dimension):
    output = torch.zeros(classes, clusters, dimension)
    valid = torch.zeros(classes, clusters, dtype=torch.bool)
    counts = torch.zeros(classes, dtype=torch.long)
    for class_index in range(classes):
        selected = [record for record in records if record[0] == class_index]
        counts[class_index] = len(selected)
        if not selected:
            continue
        vectors = F.normalize(
            torch.stack([record[1] for record in selected]),
            dim=1,
            eps=1e-6,
        )
        qualities = torch.tensor([record[2] for record in selected])
        k = min(clusters, vectors.shape[0])
        centers = [int(qualities.argmax().item())]
        while len(centers) < k:
            similarity = vectors @ vectors[centers].T
            distance = 1.0 - similarity.max(dim=1).values
            distance[centers] = -1.0
            centers.append(int(distance.argmax().item()))
        center_vectors = vectors[centers].clone()
        for _ in range(6):
            assignment = (vectors @ center_vectors.T).argmax(dim=1)
            updated = []
            for cluster_index in range(k):
                members = assignment.eq(cluster_index)
                if not members.any():
                    updated.append(center_vectors[cluster_index])
                    continue
                weights = torch.softmax(qualities[members] / 0.10, dim=0)
                updated.append(
                    F.normalize(
                        (vectors[members] * weights.unsqueeze(1)).sum(dim=0),
                        dim=0,
                        eps=1e-6,
                    )
                )
            center_vectors = torch.stack(updated)
        output[class_index, :k] = center_vectors
        valid[class_index, :k] = True
    return output, valid, counts
