import torch
import torch.nn as nn
from torch.nn.functional import embedding


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""
    def __init__(self, device):
        super(L1DistanceLoss, self).__init__()
        self.device = device
        self.word_pair_dims = (1, 2)

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.
        Args:
            predictions: A pytorch batch of predicted distances
            label_batch: A pytorch batch of true distances
            length_batch: A pytorch batch of sentence lengths
        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked -
                                                labels_masked),
                                      dim=self.word_pair_dims)
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss, total_sents


class L1DepthLoss(nn.Module):
    """Custom L1 loss for depth sequences."""
    def __init__(self, device):
        super(L1DepthLoss, self).__init__()
        self.device = device
        self.word_dim = 1

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on depth sequences.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the sentence length)
        and then across the batch.
        Args:
            predictions: A pytorch batch of predicted depths
            label_batch: A pytorch batch of true depths
            length_batch: A pytorch batch of sentence lengths
        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                total_sents: number of sentences in the batch
        """
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked -
                                                labels_masked),
                                      dim=self.word_dim)
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss, total_sents


class ContractiveLoss(nn.Module):
    """Custom contractive learning loss."""
    def __init__(self, device):
        super(ContractiveLoss, self).__init__()
        self.device = device
        self.word_dim = 1
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, embeddings, weights):
        """ Computes L1 loss on depth sequences.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the sentence length)
        and then across the batch.
        Args:
            predictions: A pytorch batch of cosine similarity score between each embedding
            similarity: A pytorch batch of initial cosine similarity score between positive examples
        Returns:
            A tuple of:
                batch_loss: average loss in the batch
                total_sents: number of sentences in the batch
        """
        predictions = self.cos(embeddings)
        predictions_diag = torch.diagonal(predictions, dim1=-2, dim2=-1)
        # normalized_diag = 1 - torch.div(torch.asb(similarity - predictions_diag), similarity)
        fraction = torch.div(torch.exp(predictions_diag),
                             torch.sum(torch.exp(predictions), dim=-1))
        loss = torch.sum(-torch.log(weights * fraction))

        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked -
                                                labels_masked),
                                      dim=self.word_dim)
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss, total_sents