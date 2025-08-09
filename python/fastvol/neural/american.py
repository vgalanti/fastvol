import torch
import torch.nn as nn
from torch import Tensor


class _VNet(nn.Module):
    """
    Neural network surrogate for American option pricing (value estimation).

    Given standard option inputs (spot, strike, c_flag, ttm, iv, rfr, div),
    this model returns the estimated American option price.

    Internally handles input scaling, feature transformations, and branching
    for call and put heads. Outputs are clipped to be non-negative via ReLU.

    This architecture is a prototype and may be modified in future iterations.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(7, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        self.c_head = nn.Sequential(
            nn.Linear(128 + 2, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

        self.p_head = nn.Sequential(
            nn.Linear(128 + 2, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    def forward(
        self,
        spot: Tensor,
        strike: Tensor,
        c_flag: Tensor,
        ttm: Tensor,
        iv: Tensor,
        rfr: Tensor,
        div: Tensor,
    ) -> Tensor:

        sk = spot / strike
        log_sk = torch.log(sk + 1e-8)
        log_ttm = torch.log(ttm + 1e-8)
        n_rfr = rfr / 0.2
        n_div = div / 0.2
        scaled_rfr = rfr * ttm / 0.4
        scaled_div = div * ttm / 0.4

        x = torch.cat(
            [
                ttm[:, None],
                log_ttm[:, None],
                iv[:, None],
                n_rfr[:, None],
                n_div[:, None],
                scaled_rfr[:, None],
                scaled_div[:, None],
            ],
            dim=1,
        )

        x = self.stem(x)
        x = torch.cat([x, sk[:, None], log_sk[:, None]], dim=1)

        cv = self.c_head(x).squeeze(-1)
        pv = self.p_head(x).squeeze(-1)
        return (cv * c_flag + pv * (1.0 - c_flag)) * strike


class _IVNet(nn.Module):
    """
    Neural network surrogate for implied volatility inversion.

    Given the observed price and standard option inputs (spot, strike, c_flag,
    ttm, rfr, div), this model estimates the corresponding implied volatility.

    Internally handles input normalization, log transformations, and separate
    heads for call and put cases. Outputs are constrained to be non-negative.

    This network is an initial version and subject to change through testing.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(8, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        self.c_head = nn.Sequential(
            nn.Linear(128 + 2, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

        self.p_head = nn.Sequential(
            nn.Linear(128 + 2, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    def forward(
        self,
        price: Tensor,
        spot: Tensor,
        strike: Tensor,
        c_flag: Tensor,
        ttm: Tensor,
        rfr: Tensor,
        div: Tensor,
    ) -> Tensor:

        vk = price / strike
        sk = spot / strike

        log_vk = torch.log(vk + 1e-8)
        log_sk = torch.log(sk + 1e-8)
        log_ttm = torch.log(ttm + 1e-8)

        n_rfr = rfr / 0.2
        n_div = div / 0.2

        scaled_rfr = rfr * ttm / 0.4
        scaled_div = div * ttm / 0.4

        x = torch.cat(
            [
                sk[:, None],
                log_sk[:, None],
                ttm[:, None],
                log_ttm[:, None],
                n_rfr[:, None],
                n_div[:, None],
                scaled_rfr[:, None],
                scaled_div[:, None],
            ],
            dim=1,
        )

        x = self.stem(x)
        x = torch.cat([x, vk[:, None], log_vk[:, None]], dim=1)

        cv = self.c_head(x).squeeze(-1)
        pv = self.p_head(x).squeeze(-1)
        out = cv * c_flag + pv * (1.0 - c_flag)

        # mask out options with 0 time value (tv)
        tv = price - torch.clamp((2 * c_flag - 1.0) * (spot - strike), min=0.0)
        out = torch.where(tv <= 1e-6, -1, out)

        return out
