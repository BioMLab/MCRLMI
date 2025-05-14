import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os
import math
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    def __init__(self, num_matrices, num_nodes, d_model):
        super(ChannelAttention, self).__init__()
        self.num_matrices = num_matrices
        self.num_nodes = num_nodes

        self.globalAvgPool = nn.AvgPool2d((d_model, self.num_nodes), (1, 1))
        # self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.num_matrices, self.num_matrices * 6)
        self.fc2 = nn.Linear(self.num_matrices * 6, self.num_matrices)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1):
        x = self.globalAvgPool(x1)
        x = x.view(x.size(0), -1)
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        z = x1 * y
        return z


class MMGCN(nn.Module):
    def __init__(self, num_features, num_matrices, num_nodes, d_model):
        super(MMGCN, self).__init__()
        self.num_matrices = num_matrices
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.conv1 = GCNConv(num_features, num_features * 2)
        self.conv2 = GCNConv(num_features * 2, d_model)
        self.channel_attention = ChannelAttention(num_matrices, num_nodes, d_model)
        self.cnn = nn.Conv2d(in_channels=self.num_matrices,
                             out_channels=d_model,
                             kernel_size=(d_model, 1),
                             stride=1,
                             bias=True)
        self.fc1 = nn.Linear(d_model, 1024)
        self.fc2 = nn.Linear(1024, d_model)

    def forward(self, feature_matrices, adj_matrices, adj_weights, d_model):
        out_list = []
        for i in range(self.num_matrices):
            x = feature_matrices[i]
            edge_index = adj_matrices[i]
            edge_weight = adj_weights[i]
            x1 = self.conv1(x, edge_index, edge_weight)
            bn = torch.nn.BatchNorm1d(num_features=x1.size(1)).to(device)
            x1 = bn(x1)
            x1 = F.relu(x1)
            x2 = self.conv2(x1, edge_index, edge_weight)
            bn = torch.nn.BatchNorm1d(num_features=x2.size(1)).to(device)
            x2 = bn(x2)
            x2 = F.relu(x2)
            x2 = x2.t()
            out_list.append(x2)
        out = torch.stack(out_list, dim=0)
        out = out.view(1, self.num_matrices, self.num_nodes, -1)
        out = self.channel_attention(out)
        out = out.view(1, self.num_matrices, -1, self.num_nodes)
        out = self.cnn(out)
        out = out.view(d_model, self.num_nodes).t()
        return out


class GCNNet(torch.nn.Module):
    def __init__(self, data, dropout, d_model):
        super(GCNNet, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.miRNA_GCN = MMGCN(data['mi_num_nodes'], data['mi_num_matrices'], data['mi_num_features'], d_model)
        self.lncRNA_GCN = MMGCN(data['lnc_num_nodes'], data['lnc_num_matrices'], data['lnc_num_features'], d_model)
        self.prediction = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.fc_1 = nn.Linear(d_model, d_model * 2)
        self.fc_2 = nn.Linear(d_model * 2, d_model)
        self.fc1 = nn.Linear(d_model * 2, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, data, d_model, mi_indices, lnc_indices, return_features=False):
        mi_x, mi_edge_index, mi_edge_weights = \
            data['mi_x'], data['mi_edge_index'], data['mi_edge_weights']
        mi_xg = self.miRNA_GCN(mi_x, mi_edge_index, mi_edge_weights, d_model).to(device)
        mi = self.dropout(mi_xg)
        mi = self.fc_1(mi)
        bn = torch.nn.BatchNorm1d(num_features=mi.size(1)).to(device)
        mi = bn(mi)
        mi = self.dropout(mi)
        mi = self.fc_2(mi)
        bn = torch.nn.BatchNorm1d(num_features=mi.size(1)).to(device)
        mi = bn(mi)

        lnc_x, lnc_edge_index, lnc_edge_weights = \
            data['lnc_x'], data['lnc_edge_index'], data['lnc_edge_weights']
        lnc_xg = self.lncRNA_GCN(lnc_x, lnc_edge_index, lnc_edge_weights, d_model).to(device)
        lnc = self.dropout(lnc_xg)
        lnc = self.fc_1(lnc)
        bn = torch.nn.BatchNorm1d(num_features=lnc.size(1)).to(device)
        lnc = bn(lnc)
        lnc = self.dropout(lnc)
        lnc = self.fc_2(lnc)

        bn = torch.nn.BatchNorm1d(num_features=lnc.size(1)).to(device)
        lnc = bn(lnc)
        mi_expanded = mi[mi_indices]
        lnc_expanded = lnc[lnc_indices]

        bn = torch.nn.BatchNorm1d(num_features=mi_expanded.size(1)).to(device)
        mi_expanded = bn(mi_expanded)
        mi_expanded = self.relu(mi_expanded)
        mi_expanded = self.dropout(mi_expanded)

        bn = torch.nn.BatchNorm1d(num_features=lnc_expanded.size(1)).to(device)
        lnc_expanded = bn(lnc_expanded)
        lnc_expanded = self.relu(lnc_expanded)
        lnc_expanded = self.dropout(lnc_expanded)
        cat = torch.cat((mi_expanded, lnc_expanded), 1)
        out1 = self.fc1(cat)

        bn = torch.nn.BatchNorm1d(num_features=out1.size(1)).to(device)
        out1 = bn(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out2 = self.fc2(out1)

        bn = torch.nn.BatchNorm1d(num_features=out2.size(1)).to(device)
        out2 = bn(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out = self.out(out2)

        bn = torch.nn.BatchNorm1d(num_features=out.size(1)).to(device)
        out_feature = bn(out)
        out = out_feature.squeeze()
        out = torch.sigmoid(out)

        if return_features:
            return mi_expanded, lnc_expanded
        else:
            return out


class valueEmbedding(nn.Module):
    def __init__(self, d_input, d_model, value_linear, value_sqrt):
        super(valueEmbedding, self).__init__()
        self.value_linear = value_linear
        self.value_sqrt = value_sqrt
        self.d_model = d_model
        self.inputLinear = nn.Linear(d_input, d_model)

    def forward(self, x):
        if self.value_linear:
            x = self.inputLinear(x)
        if self.value_sqrt:
            x = x * math.sqrt(self.d_model)
        return x


class positionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(positionalEmbedding, self).__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        return self.pos_emb[:, :x.size(1)]


class dataEmbedding(nn.Module):
    def __init__(self, d_input, d_model, value_linear, value_sqrt, posi_emb, input_dropout):
        super(dataEmbedding, self).__init__()
        self.posi_emb = posi_emb
        self.value_embedding = valueEmbedding(d_input=d_input, d_model=d_model, value_linear=value_linear,
                                              value_sqrt=value_sqrt)
        self.positional_embedding = positionalEmbedding(d_model=d_model)
        self.inputDropout = nn.Dropout(input_dropout)

    def forward(self, x):
        if self.posi_emb:
            x = self.value_embedding(x) + self.positional_embedding(x)
        else:
            x = self.value_embedding(x)
        x = self.inputDropout(x)
        return x


class scaledDotProductAttention(nn.Module):
    def __init__(self):
        super(scaledDotProductAttention, self).__init__()

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scale = 1. / math.sqrt(E)
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls, bshd->blhd", A, values)
        return V.contiguous()


class attentionLayer(nn.Module):
    def __init__(self, scaled_dot_product_attention, d_model, n_heads):
        super(attentionLayer, self).__init__()
        d_values = d_model // n_heads
        self.scaled_dot_product_attention = scaled_dot_product_attention
        self.query_linear = nn.Linear(d_model, d_values * n_heads)
        self.key_linear = nn.Linear(d_model, d_values * n_heads)
        self.value_linear = nn.Linear(d_model, d_values * n_heads)
        self.out_linear = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_linear(queries).view(B, L, H, -1)
        keys = self.key_linear(keys).view(B, S, H, -1)
        values = self.value_linear(values).view(B, S, H, -1)
        out = self.scaled_dot_product_attention(queries, keys, values)
        out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_linear(out)
        return out


class encoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, add, norm, ff, encoder_dropout):
        super(encoderLayer, self).__init__()
        d_ff = int(d_ff * d_model)
        self.attention_layer = attention_layer
        self.add = add
        self.norm = norm
        self.ff = ff
        self.feedForward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(encoder_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        new_x = self.attention_layer(x, x, x)
        if self.add:
            if self.norm:
                out1 = self.norm1(x + self.dropout(new_x))
                out2 = self.norm2(out1 + self.dropout(self.feedForward(out1)))
            else:
                out1 = x + self.dropout(new_x)
                out2 = out1 + self.dropout(self.feedForward(out1))
        else:
            if self.norm:
                out1 = self.norm1(self.dropout(new_x))
                out2 = self.norm2(self.dropout(self.feedForward(out1)))
            else:
                out1 = self.dropout(new_x)
                out2 = self.dropout(self.feedForward(out1))
        if self.ff:
            return out2
        else:
            return out1


class encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_input, d_model, n_heads, e_layers, d_ff, pos_emb, value_linear,
                 value_sqrt, add, norm, ff, dropout):
        super(Transformer, self).__init__()
        self.data_embedding = dataEmbedding(d_input, d_model,
                                            posi_emb=pos_emb,
                                            value_linear=value_linear,
                                            value_sqrt=value_sqrt,
                                            input_dropout=dropout)
        self.encoder = encoder(
            [
                encoderLayer(
                    attentionLayer(scaledDotProductAttention(), d_model, n_heads),
                    d_model, d_ff,
                    encoder_dropout=dropout,
                    add=add, norm=norm, ff=ff
                ) for l in range(e_layers)
            ]
        )

    def forward(self, x):
        x_embedding = self.data_embedding(x)
        enc_out = self.encoder(x_embedding)
        return enc_out


class TransformerNet(torch.nn.Module):
    def __init__(self, mi_seq_len1, lnc_seq_len1, mi_seq_len2, lnc_seq_len2, mi_seq_len3, lnc_seq_len3, d_input1,
                 d_input2, d_input3, d_model, dropout, num_matrices, batch_size):
        super(TransformerNet, self).__init__()
        self.num_matrices = num_matrices
        self.batch_size = batch_size
        self.d_model = d_model
        self.trans1 = Transformer(d_input=d_input1, d_model=d_model, n_heads=1, e_layers=2, d_ff=0.5,
                                  pos_emb=True, value_linear=True, value_sqrt=True, add=True, norm=True, ff=True,
                                  dropout=0.05)
        self.trans2 = Transformer(d_input=d_input2, d_model=d_model, n_heads=1, e_layers=2, d_ff=0.5,
                                  pos_emb=True, value_linear=True, value_sqrt=True, add=True, norm=True, ff=True,
                                  dropout=0.05)
        self.trans3 = Transformer(d_input=d_input3, d_model=d_model, n_heads=1, e_layers=2, d_ff=0.5,
                                  pos_emb=True, value_linear=True, value_sqrt=True, add=True, norm=True, ff=True,
                                  dropout=0.05)
        self.prediction_mi1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(mi_seq_len1 * d_model, d_model),
        )
        self.prediction_lnc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(lnc_seq_len1 * d_model, d_model),
        )
        self.prediction_mi2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(mi_seq_len2 * d_model, d_model),
        )
        self.prediction_lnc2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(lnc_seq_len2 * d_model, d_model),
        )
        self.prediction_mi3 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(mi_seq_len3 * d_model, d_model),
        )
        self.prediction_lnc3 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(lnc_seq_len3 * d_model, d_model),
        )
        self.cnn = nn.Conv2d(in_channels=self.num_matrices,
                             out_channels=d_model,
                             kernel_size=(d_model, 1),
                             stride=1,
                             bias=True)
        self.channel_attention = ChannelAttention(num_matrices, batch_size, d_model)
        self.fc1 = nn.Linear(d_model * 2, 1024)
        self.fc2 = nn.Linear(1024, d_model)
        self.out = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, mi_seq1, lnc_seq1, mi_seq2, lnc_seq2, mi_seq3, lnc_seq3, return_features=False):
        mi_xt1 = self.trans1(mi_seq1).to(device)
        lnc_xt1 = self.trans1(lnc_seq1).to(device)
        mi1 = self.prediction_mi1(mi_xt1).to(device)
        lnc1 = self.prediction_lnc1(lnc_xt1).to(device)

        mi_xt2 = self.trans2(mi_seq2).to(device)
        lnc_xt2 = self.trans2(lnc_seq2).to(device)
        mi2 = self.prediction_mi2(mi_xt2).to(device)
        lnc2 = self.prediction_lnc2(lnc_xt2).to(device)

        mi_xt3 = self.trans3(mi_seq3).to(device)
        lnc_xt3 = self.trans3(lnc_seq3).to(device)
        mi3 = self.prediction_mi3(mi_xt3).to(device)
        lnc3 = self.prediction_lnc3(lnc_xt3).to(device)

        mi_out_list = []
        mi_out_list.append(mi1)
        mi_out_list.append(mi2)
        mi_out_list.append(mi3)
        mi = torch.stack(mi_out_list, dim=0)
        mi = mi.reshape(1, self.num_matrices, -1, self.d_model)
        mi = self.channel_attention(mi)
        mi = mi.view(1, self.num_matrices, self.d_model, -1)
        mi = self.cnn(mi)
        mi = mi.view(self.d_model, -1).t()

        lnc_out_list = []
        lnc_out_list.append(lnc1)
        lnc_out_list.append(lnc2)
        lnc_out_list.append(lnc3)
        lnc = torch.stack(lnc_out_list, dim=0)
        lnc = lnc.reshape(1, self.num_matrices, -1, self.d_model)
        lnc = self.channel_attention(lnc)
        lnc = lnc.view(1, self.num_matrices, self.d_model, -1)
        lnc = self.cnn(lnc)
        lnc = lnc.view(self.d_model, -1).t()
        bn = torch.nn.BatchNorm1d(num_features=mi.size(1)).to(device)
        mi = bn(mi)
        mi = self.relu(mi)
        mi = self.dropout(mi)

        bn = torch.nn.BatchNorm1d(num_features=lnc.size(1)).to(device)
        lnc = bn(lnc)
        lnc = self.relu(lnc)
        lnc = self.dropout(lnc)
        cat = torch.cat((mi, lnc), 1)
        out1 = self.fc1(cat)

        bn = torch.nn.BatchNorm1d(num_features=out1.size(1)).to(device)
        out1 = bn(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out2 = self.fc2(out1)

        bn = torch.nn.BatchNorm1d(num_features=out2.size(1)).to(device)
        out2 = bn(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out = self.out(out2)

        bn = torch.nn.BatchNorm1d(num_features=out.size(1)).to(device)
        out_feature = bn(out)
        out = out_feature.squeeze()
        out = torch.sigmoid(out)

        if return_features:
            return mi, lnc
        else:
            return out


class KANLinear(nn.Module):
    def __init__(
            self, in_features, out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                       -1).contiguous())
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                     * self.scale_noise
                     / self.grid_size)
            self.spline_weight.data.copy_(
                self.scale_spline * self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise, ))

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
                     ) + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        spline_output = nn.functional.linear(self.b_splines(x).view(x.size(0), -1),
                                             self.spline_weight.view(self.out_features, -1), )
        return base_output + spline_output
