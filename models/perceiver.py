from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange
from models.transformer.utils import sinusoid_encoding_table

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

# helper classes Perceiver
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = x[:,:,:-remainder]  # cut the last ones to be divisible for poolsize

        attn_logits = einsum('b d n, d e -> b e n', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

import math
def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


# main classes
class Perceiver_inner(nn.Module):
    def __init__(
            self,
            *,
            depth,
            dim,
            queries_dim,
            dropout=0.2,
            logits_dim=None,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            self_per_cross_attn=1,
            final_perceiver_head=False,
            num_downsamples=4,
            att_pool_size=6,
            dim_divisible_by=2,
            predict_proteome=False
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        latent_dim_head = latent_dim // latent_heads
        latent_dim = latent_dim + 8

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads,
                                                               dim_head=cross_dim_head,dropout=dropout), context_dim=dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                                dropout=dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff,
                                                                                      get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.final_perceiver_head = final_perceiver_head
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, queries_dim)
        ) if final_perceiver_head else nn.Identity()

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                                 dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim, mult=2)) if decoder_ff else None

        # Enformer-like Query :
        half_dim = queries_dim // 2
        self.stem = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(4, queries_dim, 15, padding=7),
            Residual(ConvBlock(queries_dim)),
            AttentionPool(queries_dim, pool_size=att_pool_size)
        )
        filter_list = exponential_linspace_int(queries_dim, half_dim, num=(num_downsamples - 1), divisible_by=dim_divisible_by)
        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=att_pool_size//2)
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(queries_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        out_scores = 2 if predict_proteome else 1
        self.to_score = nn.Linear(queries_dim * 2, out_scores) if final_perceiver_head else nn.Linear(queries_dim, out_scores)

    def forward(
            self,
            data,
            mask=None,
            queries=None,
            halflife=None
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b=b)

        x = torch.cat([x, halflife.float().unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        queries = self.stem(queries)
        queries = self.conv_tower(queries)
        queries = self.flatten(queries)
        queries = torch.cat([queries, halflife.float()], dim=-1) # comment this line to exclude halflife data
        queries = self.dropout1(self.relu1(self.dense1(queries)))

        logits = self.decoder_cross_attn(queries.unsqueeze(1), context=x).squeeze(1) + queries

        if self.final_perceiver_head:
            x = self.to_logits(x)
            logits = torch.cat([logits, x], dim=-1)

        return self.to_score(logits)


class DNAPerceiver(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            max_seq_len,
            pos_emb,
            activation,
            dropout,
            depth,
            leftpos,
            rightpos,
            queries_dim,
            att_pool_size,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.dropout_emb = nn.Dropout(p=dropout)

        if pos_emb == 'sin_fix':
            self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=True)
        elif pos_emb == 'sin_learned':
            self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=False)
        else:
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver = Perceiver_inner(
                depth=depth,
                dim=dim,
                queries_dim=queries_dim,
                dropout=dropout,
                att_pool_size=att_pool_size,
                **kwargs
            )
        self.activation = activation
        self.leftpos = leftpos
        self.rightpos = rightpos

    def forward(self, halflife, x_bool, x_dna, mask=None):

        x_dna = x_dna[:, self.leftpos:self.rightpos]
        x_bool = x_bool[:, self.leftpos:self.rightpos, :]

        n, device = x_dna.shape[1], x_dna.device
        x = self.token_emb(x_dna)
        x = self.dropout_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        mask = x_dna != 0

        score = self.perceiver(x, mask=mask, queries=x_bool, halflife=halflife)

        if self.activation == 'tanh':
            return torch.tanh(score.squeeze(1))
        else:
            return score.squeeze(1)


class ProteinPerceiver_inner(nn.Module):
    def __init__(
            self,
            *,
            depth,
            dim,
            queries_dim,
            dropout=0.2,
            logits_dim=None,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            self_per_cross_attn=1,
            final_perceiver_head=False,
            num_downsamples=4,
            att_pool_size=6,
            dim_divisible_by=2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        latent_dim_head = latent_dim // latent_heads

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads,
                                                               dim_head=cross_dim_head,dropout=dropout), context_dim=dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                                dropout=dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff,
                                                                                      get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.final_perceiver_head = final_perceiver_head
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, queries_dim)
        ) if final_perceiver_head else nn.Identity()

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                                 dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim, mult=2)) if decoder_ff else None

        half_dim = queries_dim // 2
        # create stem
        self.stem = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(23, queries_dim, 15, padding=7),
            Residual(ConvBlock(queries_dim)),
            AttentionPool(queries_dim, pool_size=att_pool_size)
        )
        # create conv tower
        filter_list = exponential_linspace_int(queries_dim, half_dim, num=(num_downsamples - 1), divisible_by=dim_divisible_by)
        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=att_pool_size//2)
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(queries_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        out_scores = 1
        self.to_score = nn.Linear(queries_dim * 2, out_scores) \
            if final_perceiver_head else nn.Linear(queries_dim, out_scores)

    def forward(
            self,
            data,
            mask=None,
            queries=None
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b=b)

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        queries = self.stem(queries)
        queries = self.conv_tower(queries)
        queries = self.flatten(queries)
        queries = self.dropout1(self.relu1(self.dense1(queries)))

        logits = self.decoder_cross_attn(queries.unsqueeze(1), context=x).squeeze(1) + queries

        if self.final_perceiver_head:
            x = self.to_logits(x)
            logits = torch.cat([logits, x], dim=-1)

        return self.to_score(logits)


class ProteinPerceiver(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            max_seq_len,
            pos_emb,
            activation,
            dropout,
            depth,
            queries_dim,
            att_pool_size,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.dropout_emb = nn.Dropout(p=dropout)

        if pos_emb == 'sin_fix':
            self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=True)
        elif pos_emb == 'sin_learned':
            self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=False)
        else:
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver = ProteinPerceiver_inner(
                depth=depth,
                dim=dim,
                queries_dim=queries_dim,
                dropout=dropout,
                att_pool_size=att_pool_size,
                **kwargs
            )
        self.activation = activation
        self.max_len = max_seq_len

    def forward(self, x_seq, x_bool, mask=None):

        x_seq = x_seq[:, :self.max_len]
        x_bool = x_bool[:, :self.max_len, :]

        n, device = x_seq.shape[1], x_seq.device
        x = self.token_emb(x_seq)
        x = self.dropout_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        mask = x_seq != 0

        score = self.perceiver(x, mask=mask, queries=x_bool.float()) # queries=x

        if self.activation == 'tanh':
            return torch.tanh(score.squeeze(1))
        else:
            return score.squeeze(1)


class DNAProteinPerceiver_inner(nn.Module):
    def __init__(
            self,
            *,
            depth,
            dim,
            queries_dim,
            dropout=0.2,
            logits_dim=None,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            self_per_cross_attn=1,
            final_perceiver_head=False,
            num_downsamples=4,
            att_pool_size=6,
            dim_divisible_by=2,
            input_perc='prot',
            input_conv='dna',
            num_tokens=4,
    ):
        super().__init__()
        self.input_perc = input_perc
        self.input_conv = input_conv

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        latent_dim_head = latent_dim // latent_heads
        if self.input_perc == 'dna':
            latent_dim = latent_dim + 8 # halflife dimension to add
        if self.input_conv == 'dna':
            num_tokens_conv = 4
        else:
            num_tokens_conv = 23

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads,
                                                               dim_head=cross_dim_head,dropout=dropout), context_dim=dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                                dropout=dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff,
                                                                                      get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.final_perceiver_head = final_perceiver_head
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, queries_dim)
        ) if final_perceiver_head else nn.Identity()

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                                 dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim, mult=2)) if decoder_ff else None

        half_dim = queries_dim // 2
        # create stem
        self.stem = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(num_tokens_conv, queries_dim, 15, padding=7),
            Residual(ConvBlock(queries_dim)),
            AttentionPool(queries_dim, pool_size=att_pool_size)
        )
        # create conv tower
        filter_list = exponential_linspace_int(queries_dim, half_dim, num=(num_downsamples - 1), divisible_by=dim_divisible_by)
        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=att_pool_size//2)
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(queries_dim)  # add also halflife data
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        out_scores = 1
        self.to_score = nn.Linear(queries_dim * 2, out_scores) \
            if final_perceiver_head else nn.Linear(queries_dim, out_scores)

    def forward(
            self,
            data,
            mask=None,
            queries=None,
            halflife=None
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b=b)

        if self.input_perc == 'dna':
            x = torch.cat([x, halflife.float().unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        queries = self.stem(queries)
        queries = self.conv_tower(queries)
        queries = self.flatten(queries)
        if self.input_conv == 'dna':
            queries = torch.cat([queries, halflife.float()], dim=-1) # comment this line to exclude halflife data
        queries = self.dropout1(self.relu1(self.dense1(queries)))

        logits = self.decoder_cross_attn(queries.unsqueeze(1), context=x).squeeze(1) + queries

        if self.final_perceiver_head:
            x = self.to_logits(x)
            logits = torch.cat([logits, x], dim=-1)

        return self.to_score(logits)


class DNAProteinPerceiver(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            max_len_prot,
            max_len_dna,
            pos_emb,
            activation,
            dropout,
            depth,
            queries_dim,
            att_pool_size,
            input_perc,
            input_conv,
            leftpos,
            rightpos,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.dropout_emb = nn.Dropout(p=dropout)

        if input_perc == 'prot':
            max_seq_len = max_len_prot
        else:
            max_seq_len = max_len_dna

        if pos_emb == 'sin_fix':
            self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=True)
        elif pos_emb == 'sin_learned':
            self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=False)
        else:
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver = DNAProteinPerceiver_inner(
                depth=depth,
                dim=dim,
                queries_dim=queries_dim,
                dropout=dropout,
                att_pool_size=att_pool_size,
                input_perc=input_perc,
                input_conv=input_conv,
                num_tokens=num_tokens,
                **kwargs
            )
        self.activation = activation
        self.max_len_prot = max_len_prot
        self.max_len_dna = max_len_dna
        self.input_perc = input_perc
        self.input_conv = input_conv
        self.leftpos = leftpos
        self.rightpos = rightpos

    def forward(self, prot_seq, prot_seq_bool, dna_seq, halflife, dna_seq_bool, mask=None): #prot_seq, prot_seq_bool, dna_seq, half_life, dna_seq_bool

        if self.input_perc == 'prot':
            x_seq = prot_seq[:, :self.max_len_prot]
        else:
            x_seq = dna_seq[:, self.leftpos:self.rightpos]

        if self.input_conv == 'prot':
            x_bool = prot_seq_bool[:, :self.max_len_prot, :]
        else:
            x_bool = dna_seq_bool[:, self.leftpos:self.rightpos, :]

        n, device = x_seq.shape[1], x_seq.device
        x = self.token_emb(x_seq)
        x = self.dropout_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        mask = x_seq != 0

        score = self.perceiver(x, mask=mask, queries=x_bool.float(), halflife=halflife) # queries=x

        if self.activation == 'tanh':
            return torch.tanh(score.squeeze(1))
        else:
            return score.squeeze(1)