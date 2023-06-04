import torch
import torch.nn as nn
from torch_points3d.modules.KPConv.kernels import KPConvLayer
from torch_scatter import scatter_softmax
from timm.models.layers import DropPath, trunc_normal_
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_geometric.nn import voxel_grid
from lib.pointops2.functions import pointops

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class NSA(nn.Module):
    # neighbor self-attention 
    def __init__(self, dim, num_heads, nsample=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.nsample = nsample
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.posi_q = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_k = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_v = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        knn_idx, _ = pointops.knnquery(self.nsample, p, p, o, o) # (n, nsample)

        pos_x, _ = QueryGroup(self.nsample, p, p, x, knn_idx, o, o, local=True)  # (n, nsample, nsample, 3) (n, nsample, c)
        pos, x = pos_x
        pos = pos.reshape(pos.shape[0], self.nsample, -1)  # (n, nsample, nsample*3)

        # (n, nsample, c)
        pos_q = self.posi_q(pos)
        pos_k = self.posi_k(pos)
        pos_v = self.posi_v(pos)

        # Query, Key, Value
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (pos_q, pos_k, pos_v))

        pos_dots = torch.matmul(q, pos_q.transpose(-1, -2)) + torch.matmul(k, pos_k.transpose(-1, -2))

        dots = torch.matmul(q, k.transpose(-1, -2)) + pos_dots  # b h n n
        attn = self.attn_drop(self.softmax(dots * self.scale))

        out = torch.matmul(attn, (v + pos_v))
        out = rearrange(out, 'b h n d -> b n (h d)')

        # only pick the first one because the first neighbor is itself
        out = out[:, 0, :]  # (n, c)

        out = self.proj_drop(self.proj(out))

        return out

class LSA(nn.Module):
    # local self-attention 
    def __init__(self, dim, num_heads, nsample=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.nsample = nsample
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.posi_q = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_k = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_v = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, p_r) -> torch.Tensor:
        pos = p_r.reshape(p_r.shape[0], self.nsample, -1)

        # (n, nsample, c)
        pos_q = self.posi_q(pos)
        pos_k = self.posi_k(pos)
        pos_v = self.posi_v(pos)


        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (pos_q, pos_k, pos_v))

        pos_dots = torch.matmul(q, pos_q.transpose(-1, -2)) + torch.matmul(k, pos_k.transpose(-1, -2))

        dots = torch.matmul(q, k.transpose(-1, -2)) + pos_dots  # b h n n
        attn = self.attn_drop(self.softmax(dots * self.scale))

        out = torch.matmul(attn, (v + pos_v))

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj_drop(self.proj(out))

        return out

class NCA(nn.Module):
    # neighbor cross-attention 
    def __init__(self, dim, num_heads, nsample=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.nsample = nsample
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.posi_q = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_k = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )
        self.posi_v = nn.Sequential(
                nn.Linear(self.nsample*3, dim),
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(dim),
                Rearrange('b c n-> b n c'),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
                )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pxo, patch_pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        n_p, n_x, n_o = patch_pxo # (m, 3), (m, c), (b)

        knn_idx, _ = pointops.knnquery(self.nsample, n_p, p, n_o, o) # (n, nsample)
        pos_n_x, _ = QueryGroup(self.nsample, n_p, p, n_x, knn_idx, n_o, o, use_xyz=True)  # (n, nsample, 3+c)
        pos, n_x = pos_n_x[:, :, :3], pos_n_x[:, :, 3:] # (n, nsample, 3), (n, nsample, c)
        pos = repeat(pos, 'n m c -> n k m c', k = self.nsample)  # (n, nsample, nsample, 3)

        pos = pos.reshape(pos.shape[0], self.nsample, -1)

        # (n, nsample, c)
        pos_q = self.posi_q(pos)
        pos_k = self.posi_k(pos)
        pos_v = self.posi_v(pos)

        q, k, v = self.linear_q(x), self.linear_k(n_x), self.linear_v(n_x)  # (n, c), (n, nsample, c), (n, nsample, c)
        q = repeat(q, 'n c -> n k c', k = self.nsample)  # (n, nsample, c)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        pos_q, pos_k, pos_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (pos_q, pos_k, pos_v))

        pos_dots = torch.matmul(q, pos_q.transpose(-1, -2)) + torch.matmul(k, pos_k.transpose(-1, -2))

        dots = torch.matmul(q, k.transpose(-1, -2)) + pos_dots  # b h n n
        attn = self.attn_drop(self.softmax(dots * self.scale))

        out = torch.matmul(attn, (v + pos_v))

        out = rearrange(out, 'b h n d -> b n (h d)') # (n, nsample, c)

        # mean features because they are the same due to the repeat operation
        out = torch.mean(out, dim=1) # (n, c)

        out = self.proj_drop(self.proj(out))
        return out

def QueryGroup(nsample, xyz, new_xyz, feats, idx, offset, new_offset, use_xyz=False, local=False):
    if idx is None:
        idx, _ = pointops.knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feats.shape[1]
    knn_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz = knn_xyz - new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feats[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)

    if use_xyz:
        grouped_feat = torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)

    if local:
        grouped_xyz = knn_xyz.unsqueeze(1) - knn_xyz.unsqueeze(2) # (m, nsample, nsample, 3)
        grouped_feat = [grouped_xyz, grouped_feat]

    return grouped_feat, idx


def Sampling(xyz, offset, downscale=8):
    count = int(offset[0].item()/downscale)
    n_offset = [count]

    for i in range(1, offset.shape[0]):
        count += ((offset[i].item() - offset[i-1].item())/downscale)
        n_offset.append(count)
    n_offset = torch.tensor(n_offset, dtype=torch.int32, device=torch.device('cuda'))
    idx = pointops.furthestsampling(xyz, offset, n_offset)  # (m)
    n_xyz = xyz[idx.long(), :]  # (m, 3)
    return idx.long(), n_xyz, n_offset


class CDBlock(nn.Module):
    def __init__(self, dim, num_heads, k=16, act_layer=nn.GELU, norm_layer=nn.LayerNorm, downscale=8,
                 aggre='max', mlp_ratio=4, attn_drop=0., proj_drop=0.1, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.aggre = aggre
        self.downscale = downscale
        self.nsample = k[0]

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.l_norm1 = norm_layer(dim)
        self.l_attn = LSA(dim, num_heads, nsample=k[0], attn_drop=attn_drop, proj_drop=proj_drop)

        self.g_norm1 = norm_layer(dim)
        self.g_attn = NSA(dim, num_heads, nsample=k[1], attn_drop=attn_drop, proj_drop=proj_drop)

        self.c_norm1 = norm_layer(dim)
        self.c_norm2 = norm_layer(dim)
        self.c_attn = NCA(dim, num_heads, nsample=k[2], attn_drop=attn_drop, proj_drop=proj_drop)
        self.c_norm3 = norm_layer(dim)
        self.c_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def feats_aggre(self, feats):
        """
        feats: (m, nsample, c)
        """
        if self.aggre == 'max':
            patch_feats, _ = torch.max(feats, dim=1)
        elif self.aggre == 'mean':
            patch_feats = torch.mean(feats, dim=1)
        return patch_feats # (m, c)

    def forward(self, pxo, pko):
        p, x, o = pxo # (n, 3), (n, c), (b)
        n_p, knn_idx, n_o = pko

        identity = x.clone()

        px, _ = QueryGroup(self.nsample, p, n_p, x, knn_idx, o, n_o, local=True)  # (m, nsample, nsample, 3) (m, nsample, c)
        p_r, x  = px

        l_feat = x
        x = self.l_norm1(x)
        x = self.l_attn(x, p_r) # (m, nsample, c)
        x = l_feat + self.drop_path(x) # (m, nsample, c)

        x = self.feats_aggre(x) # (m, c)
        g_feat = x
        x = self.g_norm1(x)
        x = self.g_attn([n_p, x, n_o]) # (m, c)
        x = g_feat + self.drop_path(x)

        feat = self.c_norm1(identity) # (n, c)
        x = self.c_norm2(x)  # (m, c)
        x = self.c_attn([p, feat, o], [n_p, x, n_o]) # (n, c)
        x = identity + self.drop_path(x)
        x = x + self.drop_path(self.c_mlp(self.c_norm3(x))) # (n ,c)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, depth, num_heads, downsample=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, downscale=8, k=16,
                 aggre='max', mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.k = k
        self.nsample = k[0]
        self.downscale = downscale
        self.blocks = nn.ModuleList([
            CDBlock(in_channel, num_heads, k=k, act_layer=act_layer, norm_layer=norm_layer, downscale=downscale,
                 aggre=aggre, mlp_ratio=4, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(depth)
            ])

        self.downsample = downsample(in_channel, out_channel, self.nsample) if downsample else None

    def forward(self, pxo):
        p, x, o = pxo # (n, 3), (n, c), (b)
        sample_idx, n_p, n_o = Sampling(p, o, downscale=self.downscale)  # (m,), (m, 3), (b)
        knn_idx, _ = pointops.knnquery(self.nsample, p, n_p, o, n_o) # (m, nsample)

        for i, blk in enumerate(self.blocks):
            x = blk([p, x, o], [n_p, knn_idx, n_o])

        if self.downsample:
            feats_down, xyz_down, offset_down = self.downsample([p, x, o], [n_p, knn_idx, n_o])
        else:
            feats_down, xyz_down, offset_down = None, None, None

        feats, xyz, offset = x, p, o

        return feats, xyz, offset, feats_down, xyz_down, offset_down


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.norm = norm_layer(in_channels) if norm_layer else None
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(k)

    def forward(self, feats, xyz, offset):

        n_offset, count = [int(offset[0].item()*self.ratio)+1], int(offset[0].item()*self.ratio)+1
        for i in range(1, offset.shape[0]):
            count += ((offset[i].item() - offset[i-1].item())*self.ratio) + 1
            n_offset.append(count)
        n_offset = torch.cuda.IntTensor(n_offset)
        idx = pointops.furthestsampling(xyz, offset, n_offset)  # (m)
        n_xyz = xyz[idx.long(), :]  # (m, 3)

        feats = pointops.queryandgroup(self.k, xyz, n_xyz, feats, None, offset, n_offset, use_xyz=False)  # (m, nsample, 3+c)
        m, k, c = feats.shape
        feats = self.linear(self.norm(feats.view(m*k, c)).view(m, k, c)).transpose(1, 2).contiguous()
        feats = self.pool(feats).squeeze(-1)  # (m, c)

        return feats, n_xyz, n_offset


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.k = k
        self.norm = norm_layer(in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(k)

    def forward(self, pxo, pko):
        p, x, o = pxo
        n_p, knn_idx, n_o = pko

        feats, _ = QueryGroup(self.k, p, n_p, x, knn_idx, o, n_o, use_xyz=False)

        m, k, c = feats.shape
        feats = self.linear(self.norm(feats.view(m*k, c)).view(m, k, c)).transpose(1, 2).contiguous()
        feats = self.pool(feats).squeeze(-1)  # (m, c)

        return feats, n_p, n_o


class Upsample(nn.Module):
    def __init__(self, k, in_channels, out_channels, bn_momentum=0.02):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels))
        self.linear2 = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels))

    def forward(self, feats, xyz, support_xyz, offset, support_offset, support_feats=None):

        feats = self.linear1(support_feats) + pointops.interpolation(xyz, support_xyz, self.linear2(feats), offset, support_offset)
        return feats, support_xyz, support_offset


class KPConvSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        self.kpconv = KPConvLayer(in_channels, out_channels, point_influence=prev_grid_size * sigma, add_one=False)
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]

        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.activation(self.bn(feats))
        return feats


class KPConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_grid_size, sigma=1.0, negative_slope=0.2, bn_momentum=0.02):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(nn.Linear(in_channels, d_2, bias=False), FastBatchNorm1d(d_2, momentum=bn_momentum), activation)
        self.unary_2 = torch.nn.Sequential(nn.Linear(d_2, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum), activation)
        self.kpconv = KPConvLayer(d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False)
        # self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False), FastBatchNorm1d(out_channels, momentum=bn_momentum)
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]

        shortcut = feats
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats += shortcut
        return feats


class CDFormer(nn.Module):
    def __init__(self, downscale, depths, channels, up_k, num_heads=[1, 2, 4, 8], \
                 drop_path_rate=0.2, num_layers=4, concat_xyz=False, fea_dim=6,
                 num_classes=13, ratio=0.25, k=16, prev_grid_size=0.04, sigma=1.0, stem_transformer=False):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        k = self.tolist(k)

        if stem_transformer:
            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(fea_dim, channels[0], prev_grid_size, sigma=sigma)
            ])
            self.layer_start = 0
        else:
            self.stem_layer = nn.ModuleList([
                KPConvSimpleBlock(fea_dim, channels[0], prev_grid_size, sigma=sigma),
                KPConvResBlock(channels[0], channels[0], prev_grid_size, sigma=sigma)
            ])
            self.downsample = TransitionDown(channels[0], channels[1], ratio, k[0][0])
            self.layer_start = 1

        self.layers = nn.ModuleList([BasicBlock(channels[i], channels[i+1] if i < num_layers-1 else None, depths[i], num_heads[i], downscale=downscale,
            drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], downsample=DownSample if i < num_layers-1 else None, k=k[i]) for i in range(self.layer_start, num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.init_weights()

    def forward(self, feats, xyz, offset, batch, neighbor_idx):

        feats_stack = []
        xyz_stack = []
        offset_stack = []

        for i, layer in enumerate(self.stem_layer):
            feats = layer(feats, xyz, batch, neighbor_idx)

        feats = feats.contiguous()

        if self.layer_start == 1:
            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)
            feats, xyz, offset = self.downsample(feats, xyz, offset)

        for i, layer in enumerate(self.layers):
            feats, xyz, offset, feats_down, xyz_down, offset_down = layer([xyz, feats, offset])

            feats_stack.append(feats)
            xyz_stack.append(xyz)
            offset_stack.append(offset)

            feats = feats_down
            xyz = xyz_down
            offset = offset_down

        feats = feats_stack.pop()
        xyz = xyz_stack.pop()
        offset = offset_stack.pop()

        feats = rearrange(feats, '(b n) c -> b n c', b=len(offset))
        feats = torch.max(feats, dim=1)[0] 

        out = self.classifier(feats)

        return out

    def tolist(self, k):
        if type(k) != list:
            k = [[k]*3]*4
        return k

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

