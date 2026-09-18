##############################################################
# % Author: Castle
# % Date:01/12/2022
###############################################################

from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL1andL2
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from pointnet2_ops import pointnet2_utils


####################################################################################
# 1. 定义仅包含 Cross-Attention 的 Block
class CrossOnlyBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attn_block_style='attn', cross_attn_combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Cross Attention
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cross_attn_combine_style = cross_attn_combine_style
        cross_attn_tokens = cross_attn_block_style.split('-')
        self.cross_attn_block_length = len(cross_attn_tokens)

        self.cross_attn = None
        self.local_cross_attn = None
        for token in cross_attn_tokens:
            if token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                                 proj_drop=drop)
            elif token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                      attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                      n_group=n_group)
            elif token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)

        if self.cross_attn is not None and self.local_cross_attn is not None and cross_attn_combine_style == 'concat':
            self.cross_attn_merge_map = nn.Linear(dim * 2, dim)

    def forward(self, q, v, q_pos, v_pos, cross_attn_idx=None):
        # Cross Attention Only
        norm_q, norm_v = self.norm_q(q), self.norm_v(v)
        if self.cross_attn and self.local_cross_attn:
            feat = torch.cat([self.cross_attn(norm_q, norm_v),
                              self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)],
                             dim=-1)
            q = q + self.drop_path4(self.ls4(self.cross_attn_merge_map(feat)))
        elif self.cross_attn:
            q = q + self.drop_path4(self.ls4(self.cross_attn(norm_q, norm_v)))
        else:
            q = q + self.drop_path4(
                self.ls4(self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        # MLP
        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


# 2. 修改版 TransformerDecoder (无 self-attn)
class TransformerDecoder_Only(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attn_block_style_list=['attn'], cross_attn_combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList([CrossOnlyBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
            drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            act_layer=act_layer, norm_layer=norm_layer,
            cross_attn_block_style=cross_attn_block_style_list[i],
            cross_attn_combine_style=cross_attn_combine_style, k=k, n_group=n_group
        ) for i in range(depth)])

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for block in self.blocks:
            q = block(q, v, q_pos, v_pos, cross_attn_idx=cross_attn_idx)
        return q


# 3. 最终的 PointTransformerDecoder 接口
class PointTransformerDecoder_Only(nn.Module):
    def __init__(self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None,
                 cross_attn_block_style_list=['attn'], cross_attn_combine_style='concat', k=10, n_group=2):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerDecoder_Only(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            init_values=init_values, drop_path_rate=dpr, norm_layer=norm_layer, act_layer=act_layer,
            cross_attn_block_style_list=cross_attn_block_style_list,
            cross_attn_combine_style=cross_attn_combine_style, k=k, n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        return self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)

####################################################################################


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )


    def forward(self, x):

        return self.act(self.net2(self.net1(x)) + x)


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation1 = []
        for _ in range(blocks):
            operation1.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=1, bias=bias, activation=activation)
            )
        self.operation1 = nn.Sequential(*operation1)

        operation2 = []
        for _ in range(blocks):
            operation2.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=0.5, bias=bias, activation=activation)
            )
        self.operation2 = nn.Sequential(*operation2)

        operation3 = []
        for _ in range(blocks):
            operation3.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=1, bias=bias, activation=activation)
            )
        self.operation3 = nn.Sequential(*operation3)


        self.geo_extract = PosE_Geo(3, out_dim=6, alpha=100, beta=1000, conv_in=channels + 6, conv_out=channels)  # out_dim能被6整除  *2

    def forward(self, x, coor):  # [b, d, g]

        x_org = x
        x = self.geo_extract(coor.permute(0,2,1).unsqueeze(3), x.permute(0,2,1).unsqueeze(3)).squeeze(3)
        x = self.operation1(x)
        x = self.operation2(x)
        x = self.operation3(x).permute(0,2,1)
        x_final = x + x_org

        return x_final



class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'onebyone'
    '''

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
    ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat',
                                 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph',
                                   'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                           attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'onebyone'
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
    '''

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat',
                                           'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'

        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(
            self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph',
                                             'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                           proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                     attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                     n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat',
                                            'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'

        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(
            cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph',
                                              'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                                 proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                      attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                      n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx,
                                                           denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(
                    self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos,
                                                            idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(
                    self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos,
                                          idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos,
                                                        idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


######################################## Entry ########################################

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx)
        return x


class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
                 cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
                 k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i],
                cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx,
                      denoise_length=denoise_length)
        return q


class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x


class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list,
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list,
            cross_attn_combine_style=cross_attn_combine_style,
            k=k,
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q


class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry_Only(PointTransformerDecoder_Only):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta, conv_in, conv_out):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        self.layer = nn.Sequential(nn.Conv2d(conv_in, conv_out, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    def forward(self, knn_xyz, knn_x):  # (knn_xyz, knn_x)
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # # # Weigh
        # knn_x_w = knn_x + position_embed
        # knn_x_w *= position_embed

        connect_f = torch.cat((knn_x, position_embed), dim=1)
        knn_x_w = self.layer(connect_f) + knn_x

        return knn_x_w


######################################## Grouper ########################################
class DGCNN_Grouper(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.geo_extract1 = PosE_Geo(3, out_dim=6, alpha=100, beta=1000,
                                     conv_in=16 + 6, conv_out=16)  # out_dim能被6整除  *2

        self.geo_extract2 = PosE_Geo(3, out_dim=6, alpha=100, beta=1000,
                                     conv_in=64 + 6, conv_out=64)  # out_dim能被6整除  *2

        self.geo_extract3 = PosE_Geo(3, out_dim=6, alpha=100, beta=1000,
                                     conv_in=128 + 6, conv_out=128)  # out_dim能被6整除  *2

        self.geo_extract4 = PosE_Geo(3, out_dim=6, alpha=100, beta=1000,
                                     conv_in=128 + 6, conv_out=128)  # out_dim能被6整除  *2

        self.num_features = 128

    def fps_downsample(self, coor, x, num_group):


        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        combined_x = torch.cat([coor, x], dim=1)
        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]
        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())  # B G M
            #####################################################################################################################
            knn_xyz = index_points(coor_k.transpose(-1, -2).contiguous(), idx)
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            #####################################################################################################################
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature, knn_xyz

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128)
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f, knn_xyz = self.get_graph_feature(coor, f, coor, f)
        f = self.geo_extract1(knn_xyz, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f, knn_xyz = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.geo_extract2(knn_xyz, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f, knn_xyz = self.get_graph_feature(coor, f, coor, f)
        f = self.geo_extract3(knn_xyz, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f, knn_xyz = self.get_graph_feature(coor_q, f_q, coor, f)


        f = self.geo_extract4(knn_xyz, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class SimpleEncoder(nn.Module):
    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k

        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1]

        center = misc.fps(xyz, n_group)  # B G 3

        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'

        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()

        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size

        features = self.embedding(neighborhood)  # B G C

        return center, features


######################################## Fold ########################################
class Fold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature

        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


######################################## PCTransformer ########################################
class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        all_global_features = config.global_feature_dim
        decoder_config_stage1 = config.decoder_config_stage1
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.center_num_stage1 = getattr(config, 'center_num_stage1', [512, 128])


        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),  # 512
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )


        # Coarse Level 1 : Encoder
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )

        self.encoder = PointTransformerEncoderEntry(encoder_config)

        # Coarse Stage1
        # 获取 Transformer 层数
        self.num_layers = encoder_config.depth

        # Coarse Stage1 (为每一层实例化独立的权重)
        self.increase_dim_stage1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_config.embed_dim, all_global_features),
                nn.GELU(),
                nn.Linear(all_global_features, global_feature_dim)
            ) for _ in range(self.num_layers)
        ])

        self.coarse_stage1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_feature_dim, all_global_features),
                nn.GELU(),
                nn.Linear(all_global_features, 3 * query_num // 2)
            ) for _ in range(self.num_layers)
        ])

        self.mlp_query_stage1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_feature_dim + 3, all_global_features),
                nn.GELU(),
                nn.Linear(all_global_features, all_global_features),
                nn.GELU(),
                nn.Linear(all_global_features, decoder_config_stage1.embed_dim)
            ) for _ in range(self.num_layers)
        ])

        self.decoder_stage1 = PointTransformerDecoderEntry_Only(decoder_config_stage1)


        # Coarse Stage2
        self.increase_dim_stage2 = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, all_global_features),
            nn.GELU(),
            nn.Linear(all_global_features, global_feature_dim))

        self.coarse_stage2 = nn.Sequential(
            nn.Linear(global_feature_dim, all_global_features),
            nn.GELU(),
            nn.Linear(all_global_features, 3 * query_num)     # 3 * query_num
        )


        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)

        # Coarse Level 2 : Decoder
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, all_global_features),
            nn.GELU(),
            nn.Linear(all_global_features, all_global_features),
            nn.GELU(),
            nn.Linear(all_global_features, decoder_config.embed_dim)
        )

        self.mlp_query_noise = nn.Sequential(
            nn.Linear(global_feature_dim + 3, all_global_features),
            nn.GELU(),
            nn.Linear(all_global_features, all_global_features),
            nn.GELU(),
            nn.Linear(all_global_features, decoder_config.embed_dim)
        )


        self.decoder = PointTransformerDecoderEntry(decoder_config)

        self.norm_f_stage2 = nn.LayerNorm(encoder_config.embed_dim)


        self.reduce_stage2 = nn.Sequential(
            nn.Linear(encoder_config.embed_dim * 2, encoder_config.embed_dim),
            nn.GELU(),
            nn.Linear(encoder_config.embed_dim, encoder_config.embed_dim)
        )

        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def forward(self, xyz):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        coor_stage1, f_stage1 = self.grouper(xyz, self.center_num_stage1)  # b n c

        x = self.input_proj(f)
        x_stage1 = self.input_proj(f_stage1)
        x_stage1_org = x_stage1

        #################################################################################################################
        pe = self.pos_embed(coor)
        pe_stage1 = self.pos_embed(coor_stage1)
        x = x + pe
        x_stage1 = x_stage1 + pe_stage1

        coarse_stage1_list = []  # 【新增 1】建立一个空列表，用于收集每一层的粗糙点云


        num_layers = len(self.encoder.blocks.blocks)
        for i in range(num_layers):
            # 1. x 和 x_stage1 分别前向经过对应第 i 层的 Encoder Block
            # (深入调用 .blocks.blocks[i] 来单步执行)
            x = self.encoder.blocks.blocks[i](x, coor)
            x_stage1 = self.encoder.blocks.blocks[i](x_stage1, coor_stage1,)

            # 2. 根据当前层的 x_stage1 计算全局特征和粗糙点 coarse_stage1
            global_feature_stage1 = self.increase_dim_stage1[i](x_stage1)  # B N C
            global_feature_stage1 = torch.max(global_feature_stage1, dim=1)[0]  # B C
            coarse_stage1 = self.coarse_stage1[i](global_feature_stage1).reshape(bs, -1, 3)

            coarse_stage1_list.append(coarse_stage1)  # 【新增 2】把当前层的点云加到列表中


            # 3. 动态生成当前层的 query
            q_stage1 = self.mlp_query_stage1[i](
                torch.cat([
                    global_feature_stage1.unsqueeze(1).expand(-1, coarse_stage1.size(1), -1),
                    coarse_stage1], dim=-1))  # b n c

            x_stage2 = self.decoder_stage1.blocks.blocks[i](
                q=q_stage1, v=x,
                q_pos=coarse_stage1, v_pos=coor
            )
            x_stage1 = x_stage2

        x = x_stage1
        #################################################################################################################

        global_feature_stage2 = self.increase_dim_stage2(x)  # B N 1024
        global_feature_stage2 = torch.max(global_feature_stage2, dim=1)[0]  # B 1024
        coarse_stage2 = self.coarse_stage2(global_feature_stage2).reshape(bs, -1, 3)


        coarse_inp = misc.fps(xyz, self.num_query//2)  # B 128 3  self.num_query // 2
        coarse_stage2x = torch.cat([coarse_stage2, coarse_inp], dim=1)  # B 224+128 3?
        # # query selection
        query_ranking = self.query_ranking(coarse_stage2x)  # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
        coarse = torch.gather(coarse_stage2x, 1, idx[:, :self.num_query].expand(-1, -1, coarse_stage2x.size(-1)))


        mem = self.mem_link(x)
        global_feature = global_feature_stage2

############################################################################################################

        if self.training:

            global_feature1 = global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1)
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature1,
                    coarse], dim=-1))  # b n c


            query_xyz = coarse.contiguous()  # B, M, 3 (当前生成的点)
            known_xyz = coor_stage1.contiguous()  # B, N, 3 (Stage1 残缺输入的点)
            known_feat = x_stage1_org.transpose(1, 2).contiguous()  # B, C, N (Stage1 特征)
            # 使用 pointnet2_ops 提供的 CUDA 算子加速
            dist, idx = pointnet2_utils.three_nn(query_xyz, known_xyz)  # B, M, 3
            dist_recip = 1.0 / torch.clamp(dist, min=1e-10)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # B, M, 3
            f_stage2 = pointnet2_utils.three_interpolate(known_feat, idx, weight)
            f_stage2 = f_stage2.transpose(1, 2).contiguous()  # 转回 B, M, C


            f_stage2_aligned = self.norm_f_stage2(f_stage2)
            q_mixed = torch.cat([q, f_stage2_aligned], dim=-1)
            q_res = self.reduce_stage2(q_mixed)
            q_pure = q + q_res


            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            denoise_length = 64   # None 64


            global_feature2 = global_feature.unsqueeze(1).expand(-1, picked_points.size(1), -1)
            # produce query
            q = self.mlp_query_noise(
                torch.cat([
                    global_feature2,
                    picked_points], dim=-1))  # b n c


##################################################################################################################################
            q = torch.cat([q_pure, q], dim=1)  # (B, M+64, C)
            coarse = torch.cat([coarse, picked_points], dim=1)  # (B, M+64, 3)

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor_stage1, denoise_length=denoise_length)

##################################################################################################################################
            return q, coarse, coarse_stage2, coarse_stage1_list, denoise_length

        else:

            global_feature1 = global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1)
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature1,
                    coarse], dim=-1))  # b n c

            query_xyz = coarse.contiguous()  # B, M, 3 (当前生成的点)
            known_xyz = coor_stage1.contiguous()  # B, N, 3 (Stage1 残缺输入的点)
            known_feat = x_stage1_org.transpose(1, 2).contiguous()  # B, C, N (Stage1 特征)
            # 使用 pointnet2_ops 提供的 CUDA 算子加速
            dist, idx = pointnet2_utils.three_nn(query_xyz, known_xyz)  # B, M, 3
            dist_recip = 1.0 / torch.clamp(dist, min=1e-10)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # B, M, 3
            f_stage2 = pointnet2_utils.three_interpolate(known_feat, idx, weight)
            f_stage2 = f_stage2.transpose(1, 2).contiguous()  # 转回 B, M, C

            f_stage2_aligned = self.norm_f_stage2(f_stage2)
            q_mixed = torch.cat([q, f_stage2_aligned], dim=-1)
            q_res = self.reduce_stage2(q_mixed)
            q_pure = q + q_res  # 纯净 Query 进化完成


##################################################################################################################################
            q = q_pure

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor_stage1)

##################################################################################################################################
            return q, coarse, coarse_stage2, coarse_stage1_list, 0


######################################## MsComplete ########################################
@MODELS.register_module()
class MsComplete(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        all_global_features = config.global_feature_dim # 1024 or 512
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 4
        self.base_model = PCTransformer(config)

        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2,
                                                        step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)



        self.increase_dim_stage3 = nn.Sequential(
            nn.Conv1d(self.trans_dim, all_global_features, 1),
            nn.BatchNorm1d(all_global_features),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(all_global_features, all_global_features, 1)
        )



        self.reduce_map_stage3 = nn.Linear(self.trans_dim + all_global_features + 3, self.trans_dim)


        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        # self.loss_func = ChamferDistanceL2()



    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, coarse_stage2, coarse_stage1_list, denoised_coarse, denoised_fine, pred_fine = ret
        assert pred_fine.size(1) == gt.size(1)

        # # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k
        denoised_target = index_points(gt, idx)  # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # 【新增代码段】遍历计算 6 次 coarse_stage1 的 Loss
        loss_stage1 = 0.0
        for coarse_s1 in coarse_stage1_list:
            loss_stage1 += self.loss_func(coarse_s1, gt)
        # 平均一下，防止这 6 层加起来产生的数值过大主导了总 loss
        loss_stage1 = loss_stage1 / len(coarse_stage1_list)


        loss_coarse = 0.34 * self.loss_func(pred_coarse, gt) + 0.34 * self.loss_func(coarse_stage2, gt) + 0.34 * loss_stage1

        # recon loss
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine


        return loss_denoised, loss_recon

    def forward(self, xyz):
        q, coarse_point_cloud, coarse_stage2, coarse_stage1_list, denoise_length = self.base_model(xyz)  # B M C and B M 3

        B, M, C = q.shape

        global_feature = self.increase_dim_stage3(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        # NOTE: foldingNet
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map_stage2(rebuild_feature.reshape(B * M, -1))  # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map_stage3(rebuild_feature)  # B M C
            relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            if denoise_length == None:
                pred_fine = rebuild_points.reshape(B, -1, 3).contiguous()
                pred_coarse = coarse_point_cloud.contiguous()

                denoised_fine = rebuild_points.reshape(B, -1, 3).contiguous()
                denoised_coarse = coarse_point_cloud.contiguous()

            else:
                pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
                pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

                denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
                denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, coarse_stage2, coarse_stage1_list, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret