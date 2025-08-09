import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Callable


class _Dataset(Dataset):
    """
    Dataset for training neural network surrogates for option pricing.

    Generates synthetic option data (spot, ttm, iv, etc.) and computes corresponding
    target prices using a high-accuracy ground truth pricing function (e.g. BOPM with
    many steps, PSOR, or another trusted method).

    All inputs and outputs are filtered to exclude NaNs/Infs and converted to tensors.

    Parameters
    ----------
    fp64_truth : Callable
        Callable that takes in (spot, strike, c_flag, ttm, iv, rfr, div) and returns
        the true American option price. Must return a NumPy array of shape (n_samples,).
    n_samples : int, optional
        Number of synthetic samples to generate. Default is 100,000.
    device : str or torch.device, optional
        Device on which to allocate tensors. Default is "cpu".
    """

    def __init__(
        self,
        fp64_truth: Callable,
        n_samples: int = 100_000,
        device="cpu",
        for_iv_net: bool = False,
    ):

        def uniform(n: int, lo: float, hi: float):
            return torch.rand((n,), dtype=torch.float64, device="cpu") * (hi - lo) + lo

        # split evenly between uniformly sampled over the grid
        # and steeper low-iv low-ttm regions
        k = n_samples // 2

        strike = torch.full((n_samples,), 100.0, dtype=torch.float64)
        spot = uniform(n_samples, 30.0, 200.0)
        c_flag = torch.randint(0, 2, (n_samples,), dtype=torch.uint8)
        ttm = uniform(n_samples, 0.001, 2.0)
        iv = uniform(n_samples, 0.01, 2.0)
        rfr = uniform(n_samples, 0.0, 0.15)
        div = uniform(n_samples, 0.0, 0.10)

        sk = 0.75 + 0.5 * torch.distributions.Beta(5.0, 5.0).sample((k,))
        spot[-k:] = sk * 100.0
        ttm[-k:] = torch.exp(torch.empty(k).uniform_(np.log(1e-3), np.log(0.2)))
        iv[-k:] = torch.exp(torch.empty(k).uniform_(np.log(1e-2), np.log(0.3)))

        # compute ground truth prices
        price = fp64_truth(spot, strike, c_flag, ttm, iv, rfr, div)

        # mask out any invalid cases
        m = ~(torch.isnan(price) | torch.isinf(price) | (price <= 0) | (price > 250))

        # if for iv training, mask out values with 0 time value
        if for_iv_net:
            tv = price - torch.clamp((2 * c_flag - 1.0) * (spot - strike), min=0.0)
            m = m & (tv > 1e-6)

        self.n_samples = m.sum()
        self.spot = spot[m].clone().to(device=device)
        self.strike = strike[m].clone().to(device=device)
        self.c_flag = c_flag[m].clone().to(device=device, dtype=torch.float64)
        self.ttm = ttm[m].clone().to(device=device)
        self.iv = iv[m].clone().to(device=device)
        self.rfr = rfr[m].clone().to(device=device)
        self.div = div[m].clone().to(device=device)
        self.price = price[m].clone().to(device=device)

    def __len__(self):
        """Return number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Returns
        -------
        tuple of Tensors
            (spot, strike, c_flag, ttm, iv, rfr, div, price)
        """
        return (
            self.spot[idx],
            self.strike[idx],
            self.c_flag[idx],
            self.ttm[idx],
            self.iv[idx],
            self.rfr[idx],
            self.div[idx],
            self.price[idx],
        )


def _train_model(
    model: nn.Module,
    dataset_fn: Callable[[str], Dataset],
    loss_fn: Callable,
    predict_fn: Callable,
    target_fn: Callable,
    epochs: int = 500,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    eval_fn: Callable | None = None,
):
    """
    Core training loop for a surrogate model using synthetic data.

    Runs SGD over multiple epochs using a specified data generation function,
    loss function, and callable prediction/target interfaces.

    Parameters
    ----------
    model : nn.Module
        Neural network to be trained.
    dataset_fn : Callable[[torch.dtype, str], Dataset]
        Function that returns a Dataset instance given dtype and device.
    loss_fn : Callable
        Loss function to optimize (e.g. nn.MSELoss()).
    predict_fn : Callable
        Function that maps (model, *batch) to prediction tensor.
    target_fn : Callable
        Function that maps (*batch) to ground truth tensor.
    epochs : int, optional
        Number of training epochs. Default is 500.
    batch_size : int, optional
        Batch size for training. Default is 2048.
    lr : float, optional
        Learning rate. Default is 1e-3.
    dtype : torch.dtype, optional
        Floating-point precision used for training.
    device : {"cpu", "cuda", "mps"}, optional
        Device to train on.
    eval_fn : Callable or None, optional
        Optional evaluation function to call after training completes.
    """
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        if epoch % 20 == 0:
            ds = dataset_fn(device=device)
            dl = DataLoader(ds, batch_size=batch_size)

        total_loss = 0.0
        for batch in dl:
            optim.zero_grad()
            pred = predict_fn(model, *batch)
            target = target_fn(*batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optim.step()
            total_loss += loss.item() * pred.shape[0]

        print(f"Epoch {epoch+1:>4}: loss = {total_loss / len(dl.dataset):.6f}")

    # Optional final evaluation
    if eval_fn is not None:
        eval_fn(model, dataset_fn)


def train_v(
    model,
    fp64_truth,
    epochs: int = 500,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    n_samples_per_20_epochs: int = 100_000,
    batch_size: int = 2048,
    lr: float = 1e-3,
):
    """
    Train a neural surrogate model to estimate American option prices (vnet).

    Generates synthetic training data using `fp64_truth` as the ground truth
    pricing function. Uses a fixed strike and random values for other inputs.

    Parameters
    ----------
    model : nn.Module
        Neural network to be trained.
    fp64_truth : Callable
        High-accuracy pricing function to generate training targets.
    epochs : int, optional
        Number of training epochs. Default is 500.
    device : {"cpu", "cuda", "mps"}, optional
        Device on which to train. Default is "cpu".
    n_samples_per_20_epochs : int, optional
        Number of synthetic training samples to generate every 20 epochs.
    batch_size : int, optional
        Batch size used during training. Default is 2048.
    lr : float, optional
        Learning rate. Default is 1e-3.
    """

    def dataset_fn(device):
        return _Dataset(
            fp64_truth,
            n_samples=n_samples_per_20_epochs,
            device=device,
        )

    def predict_fn(m, *args):
        return m(*args[:7])

    def target_fn(*args):
        return args[7]

    def _eval_v(model, dataset_fn):
        model.eval()
        ds = dataset_fn(device=device)
        dl = DataLoader(ds, batch_size=25)

        with torch.no_grad():
            for spot, strike, c_flag, ttm, iv, rfr, div, price in dl:
                pred = model(spot, strike, c_flag, ttm, iv, rfr, div)
                err = torch.abs(pred - price)
                print(
                    f"\n spot   | strike | c/p | ttm  | iv   |  pred   v  true   | err "
                )
                print("--------------------------------------------------------------")
                for i, p in enumerate(pred):
                    print(
                        f" {spot[i]:6.2f} | 100.0  |  {'c' if c_flag[i] else 'p'}  | {ttm[i]:4.2f} | {iv[i]:4.2f} |  {p:5.2f}  v  {price[i]:5.2f}  | {err[i]:4.2f}"
                    )
                break

    _train_model(
        model,
        dataset_fn,
        nn.MSELoss(),
        predict_fn,
        target_fn,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        eval_fn=_eval_v,
    )


def train_iv(
    model,
    fp64_truth,
    epochs: int = 500,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    n_samples_per_20_epochs: int = 100_000,
    batch_size: int = 2048,
    lr: float = 1e-3,
):
    """
    Train a neural surrogate model to estimate implied volatility (ivnet).

    Uses synthetic input data and a high-accuracy pricing function (`fp64_truth`)
    to compute forward prices, which are then inverted by the model to recover IV.

    Parameters
    ----------
    model : nn.Module
        Neural network to be trained.
    fp64_truth : Callable
        Pricing function used to generate true prices (inputs to the model).
    epochs : int, optional
        Number of training epochs. Default is 500.
    device : {"cpu", "cuda", "mps"}, optional
        Device on which to train. Default is "cpu".
    n_samples_per_20_epochs : int, optional
        Number of synthetic training samples to generate every 20 epochs.
    batch_size : int, optional
        Batch size used during training. Default is 2048.
    lr : float, optional
        Learning rate. Default is 1e-3.
    """

    def dataset_fn(device):
        return _Dataset(
            fp64_truth,
            n_samples=n_samples_per_20_epochs,
            for_iv_net=True,
            device=device,
        )

    def predict_fn(m, *args):
        return m(args[7], *args[:4], *args[5:7])

    def target_fn(*args):
        return args[4]

    def _eval_iv(model, dataset_fn):
        model.eval()
        ds = dataset_fn(device=device)
        dl = DataLoader(ds, batch_size=25)

        with torch.no_grad():
            for spot, strike, c_flag, ttm, iv, rfr, div, price in dl:
                pred = model(price, spot, strike, c_flag, ttm, rfr, div)
                err = torch.abs(pred - iv)

                print(
                    f"\n price  | spot   | strike | c/p | ttm  |   pred    v    true   | err "
                )
                print(
                    "-----------------------------------------------------------------------"
                )
                for i, p in enumerate(pred):
                    print(
                        f" {price[i]:6.2f} | {spot[i]:6.2f} | 100.0  |  {'c' if c_flag[i] else 'p'}  | {ttm[i]:4.2f} |  {p*100:6.2f}%  v  {iv[i]*100:6.2f}%  | {err[i]*100:5.2f}"
                    )
                break

    return _train_model(
        model,
        dataset_fn,
        nn.MSELoss(),
        predict_fn,
        target_fn,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        eval_fn=_eval_iv,
    )
