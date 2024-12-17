import torch
from torch import nn


class TripletLogitsLoss(nn.Module):
    def __init__(self, args):
        super(TripletLogitsLoss, self).__init__()
        self.args = args

    def forward(self, query_embeddings, pos_embeddings, neg_embeddings):
        # [batch_size, embedding_dim] -> [batch_size]
        sim_pos_mat = torch.einsum("bd,bd->b", query_embeddings, pos_embeddings)
        sim_pos_mat = sim_pos_mat / self.args.temperature

        # [batch_size, embedding_dim] -> [batch_size, batch_size]
        sim_neg_mat = torch.einsum("bd,cd->bc", query_embeddings, neg_embeddings)
        sim_neg_mat = sim_neg_mat / self.args.temperature

        sim_diff_mat = sim_pos_mat.unsqueeze(1) - sim_neg_mat
        loss = -torch.log(torch.sigmoid(sim_diff_mat)).mean()

        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, args):
        super(InfoNCELoss, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_embeddings, pos_embeddings, neg_embeddings):
        # [batch_size, embedding_dim] -> [batch_size]
        sim_pos_mat = torch.einsum("bd,bd->b", query_embeddings, pos_embeddings)
        sim_pos_mat = sim_pos_mat / self.args.temperature

        # [batch_size, embedding_dim] -> [batch_size, batch_size]
        sim_neg_mat = torch.einsum("bd,cd->bc", query_embeddings, neg_embeddings)
        sim_neg_mat = sim_neg_mat / self.args.temperature

        logits = torch.cat([sim_pos_mat.unsqueeze(1), sim_neg_mat], dim=1)
        labels = torch.zeros(logits.shape[0]).long().to(logits.device)
        loss = self.criterion(logits, labels)

        return loss
