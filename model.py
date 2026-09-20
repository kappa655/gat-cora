import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CoraGAT(torch.nn.Module):

    def __init__(
        self,
        num_features,
        hidden_features,
        out_features,
        num_head
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels=num_features,
            out_channels=hidden_features,
            heads=num_head,
            dropout=0.6
        )

        self.conv2 = GATConv(
            in_channels=hidden_features * num_head,
            out_channels=out_features,
            heads=1,
            concat=False,
            dropout=0.4
        )

    def forward(
        self,
        input_feature,
        edge_index,
        return_att=False,
        return_embedding=False
    ):

        x = F.dropout(
            input_feature,
            p=0.6,
            training=self.training
        )

        if return_att:
            x, (att_edge_index, alpha) = self.conv1(
                x,
                edge_index,
                return_attention_weights=True
            )
        else:
            x = self.conv1(
                x,
                edge_index
            )

        x = F.elu(x)

        embedding = x

        x = F.dropout(
            x,
            p=0.6,
            training=self.training
        )

        out = self.conv2(
            x,
            edge_index
        )

        if return_att and return_embedding:
            return out, embedding, att_edge_index, alpha

        if return_att:
            return out, att_edge_index, alpha

        if return_embedding:
            return out, embedding

        return out