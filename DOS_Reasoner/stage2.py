import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from DOS_Reasoner.transformer import (
    Decoder, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward,
)


class InitNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.label_dim = args.label_dim
        self.dec_layers = args.dec_layers
        self.dec_dropout = args.dec_dropout
        self.d_model = args.d_model
        self.h = args.h
        self.d_ff = args.d_ff
        self.ctx_dim = args.d_model * 2

        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dec_dropout)

        self.tgt_embed = nn.Embedding(self.label_dim, self.d_model)
        self.fc = nn.Linear(self.d_model + 1 + self.ctx_dim, self.d_model)
        self.decoder1 = Decoder(
            DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout),
            self.dec_layers,
        )
        self.decoder2 = Decoder(
            DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout),
            self.dec_layers,
        )
        self.generator = nn.Linear(self.d_model, 1)

    def forward(self, p0, crystal_ctx):
        bs = p0.shape[0]
        device = p0.device
        pos = torch.arange(self.label_dim, device=device).unsqueeze(0).repeat(bs, 1)
        pos_emb = self.tgt_embed(pos)
        p0_feat = p0.unsqueeze(-1)
        ctx_rep = crystal_ctx.unsqueeze(1).repeat(1, self.label_dim, 1)
        x = torch.cat([pos_emb, p0_feat, ctx_rep], dim=-1)
        x = F.leaky_relu(self.fc(x))
        x = self.decoder1(x, x, None, None, flag=True)
        x = self.decoder2(x, x, None, None, flag=True)
        return self.generator(x).squeeze(-1)


class VelocityNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.label_dim = args.label_dim
        self.dec_layers = args.dec_layers
        self.dec_dropout = args.dec_dropout
        self.d_model = args.d_model
        self.h = args.h
        self.d_ff = args.d_ff
        self.ctx_dim = args.d_model * 2

        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dec_dropout)

        self.tgt_embed = nn.Embedding(self.label_dim, self.d_model)
        # per-bin input: pos_emb + p_t + t_frac + crystal_ctx
        self.fc = nn.Linear(self.d_model + 1 + 1 + self.ctx_dim, self.d_model)
        self.decoder1 = Decoder(
            DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout),
            self.dec_layers,
        )
        self.decoder2 = Decoder(
            DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout),
            self.dec_layers,
        )
        self.generator = nn.Linear(self.d_model, 1)
        # Zero-init -> velocity is 0 at start -> upwind iteration is identity.
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def forward(self, p_t, t_frac, crystal_ctx):
        bs = p_t.shape[0]
        device = p_t.device
        pos = torch.arange(self.label_dim, device=device).unsqueeze(0).repeat(bs, 1)
        pos_emb = self.tgt_embed(pos)
        pt_feat = p_t.unsqueeze(-1)
        t_feat = torch.full((bs, self.label_dim, 1), float(t_frac), device=device)
        ctx_rep = crystal_ctx.unsqueeze(1).repeat(1, self.label_dim, 1)
        x = torch.cat([pos_emb, pt_feat, t_feat, ctx_rep], dim=-1)
        x = F.leaky_relu(self.fc(x))
        x = self.decoder1(x, x, None, None, flag=True)
        x = self.decoder2(x, x, None, None, flag=True)
        return self.generator(x).squeeze(-1)


# ============================================================================
# Mass-Conserving Iterative Refinement
# ============================================================================
class CPPR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.K = int(getattr(args, "k_steps", 5))
        self.dt_step = 1.0 / self.K
        self.x_min = -4.0
        self.x_max = 4.0
        self.num_points = args.label_dim
        self.dx = (self.x_max - self.x_min) / (self.num_points - 1)

        # CFL clamp for the explicit upwind transport in `_mir_step`.
        # The per-bin Courant number is c_i = v_i * dt / dx. For the upwind
        # density update d_i^{n+1} = (1 - c_i^+ - c_{i-1}^-) d_i
        #                            + c_i^- d_{i+1} + c_{i-1}^+ d_{i-1}
        # the cross terms are >= 0, so nonnegativity needs the self-coefficient
        # 1 - c_i^+ - c_{i-1}^- >= 0. A sign-varying per-bin velocity can drive
        # this to 1 - 1 - 1 = -1 at a diverging interface, so the usual |c|<=1
        # linear-stability bound is insufficient. Requiring |c_i| <= 1/2
        # guarantees the self-coefficient >= 0 (discrete maximum principle),
        # hence non-negative densities. Set cfl_courant=1.0 for the looser
        # linear-stability-only bound.
        self.cfl_courant = float(getattr(args, "cfl_courant", 0.5))
        self.v_max = self.cfl_courant * self.dx / self.dt_step

        # Stage-2a: residual head (zero-init for identity start).
        self.residual_net = InitNet(args)
        nn.init.zeros_(self.residual_net.generator.weight)
        nn.init.zeros_(self.residual_net.generator.bias)

        # Stage-2b: velocity head.
        self.velocity_net = VelocityNet(args)

    def _residual_step(self, p, crystal_ctx):
        delta = self.residual_net(p, crystal_ctx)
        logits = torch.log(p.clamp(min=1e-10)) + delta
        return F.softmax(logits, dim=1)

    def _mir_step(self, cdf, p_t, t_frac, crystal_ctx):
        v = self.velocity_net(p_t, t_frac, crystal_ctx)  # [bs, N]
        # Enforce CFL: |v| * dt/dx <= cfl_courant (<= 1/2 => non-negative
        # upwind density update, see __init__). Without this the velocity is
        # unbounded and the downstream .clamp(min=1e-10) silently breaks the
        # mass conservation CPPR relies on.
        v = v.clamp(-self.v_max, self.v_max)
        boundary = torch.ones_like(v)
        boundary[:, 0] = 0.0
        boundary[:, -1] = 0.0
        v = v * boundary
        diff_left = torch.zeros_like(cdf)
        diff_left[:, 1:] = (cdf[:, 1:] - cdf[:, :-1]) / self.dx
        diff_right = torch.zeros_like(cdf)
        diff_right[:, :-1] = (cdf[:, 1:] - cdf[:, :-1]) / self.dx
        mask_pos = (v > 0).float()
        dF_dx = mask_pos * diff_left + (1 - mask_pos) * diff_right
        return (cdf - self.dt_step * v * dF_dx).clamp(0.0, 1.0)

    def _full_forward(self, p_0, crystal_ctx):
        # Stage-2a: residual reweighting.
        p_mid = self._residual_step(p_0, crystal_ctx)
        # Stage-2b: K CPPR transport steps.
        cdf = torch.cumsum(p_mid, dim=1)
        cdf = cdf / cdf[:, -1:].clamp(min=1e-6)
        for k in range(self.K):
            t_frac = k / self.K
            cdf_pad = F.pad(cdf, (1, 0), mode="constant", value=0)
            p_t = (cdf_pad[:, 1:] - cdf_pad[:, :-1]).clamp(min=1e-10)
            p_t = p_t / p_t.sum(dim=1, keepdim=True)
            cdf = self._mir_step(cdf, p_t, t_frac, crystal_ctx)
        cdf_pad = F.pad(cdf, (1, 0), mode="constant", value=0)
        p_final = (cdf_pad[:, 1:] - cdf_pad[:, :-1]).clamp(min=1e-10)
        return p_final / p_final.sum(dim=1, keepdim=True)

    def forward(self, x_mass, y_mass, crystal_ctx):
        p_final = self._full_forward(x_mass, crystal_ctx)
        log_target = (y_mass + 1e-10).log()
        log_pred = (p_final + 1e-10).log()
        return torch.sum(y_mass * (log_target - log_pred), dim=1).mean()

    @torch.no_grad()
    def inference(self, p0_mass, crystal_ctx):
        return self._full_forward(p0_mass, crystal_ctx)
