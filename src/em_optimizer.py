from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import (
    TorchInputLayer,
    TorchSumLayer,
    TorchGaussianLayer,
    TorchCategoricalLayer,
    TorchCPTLayer,
    TorchLayer,
    TorchHadamardLayer,
)
from torch import Tensor
import torch


from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional


def normalize_weight(circuit: TorchCircuit):
    with torch.no_grad():
        for layer_idx, layer in enumerate(circuit.nodes):
            # Sum layer
            if isinstance(layer, TorchSumLayer) or isinstance(layer, TorchCPTLayer):
                weight = list(layer.parameters())[0]
                weight.div_(weight.sum(dim=-1, keepdim=True))
            elif isinstance(layer, TorchCategoricalLayer):
                probs = list(layer.parameters())[0]
                probs.div_(probs.sum(dim=-1, keepdim=True))


def reinit_weight_pyjuice(circuit: TorchCircuit, perturbation=2.0):
    print("initing weight with Pyjuice initialization")
    with torch.no_grad():
        for layer_idx, layer in enumerate(circuit.nodes):
            # Sum layer
            if isinstance(layer, TorchSumLayer) or isinstance(layer, TorchCPTLayer):
                weight = list(layer.parameters())[0]
                weight.data = torch.exp(torch.rand(weight.shape) * -perturbation)
                weight.div_(weight.sum(dim=-1, keepdim=True))
            elif isinstance(layer, TorchCategoricalLayer):
                probs = list(layer.parameters())[0]
                probs.data = torch.exp(torch.rand(probs.shape) * -perturbation)
                probs.div_(probs.sum(dim=-1, keepdim=True))


class TorchCircuitEMOptimizer(Optimizer):
    """EM-style optimizer for a TorchCircuit."""

    def __init__(
        self,
        pc,  # TorchCircuit instance
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        alpha: float = 1e-8,
        grad_acc=0,
    ):
        if not 0.0 <= lr <= 1.0:
            raise ValueError("lr must be in [0,1]")
        defaults = dict(lr=lr, alpha=alpha)
        super().__init__(params, defaults)
        self.pc = pc
        self._log_likelihoods = {}
        self._accumulator_categorical = {}
        self.grad_acc = grad_acc
        self._current_acc_step = 0

    def log_likelihood(self, batch):
        return self.pc.evaluate(batch, module_fn=self.em_forward)

    def em_forward(self, layer, *inputs):
        # record per-layer outputs (assumes log-semiring on input layer)
        if isinstance(layer, TorchInputLayer):
            inputs = inputs[0]
            output = layer(inputs)
            output.retain_grad()
            self._log_likelihoods[layer] = (inputs, output)
        else:
            output = layer(*inputs)
        return output

    @torch.no_grad()
    def em_backward(self, layer):
        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        # if list(layer.parameters())[0].isnan().any():
        #     import pdb
        #
        #     pdb.set_trace()

        if len(self._log_likelihoods) == 0:
            return

        if isinstance(layer, TorchSumLayer) or isinstance(layer, TorchCPTLayer):
            if self.grad_acc == 0 or self.grad_acc == self._current_acc_step:
                weight = list(layer.parameters())[0]
                old_weight = weight.detach().clone()
                weight.mul_(weight.grad).clamp_(min=alpha)
                weight.div_(weight.sum(dim=-1, keepdim=True))
                weight.data = old_weight.lerp(weight, weight=lr)

        elif isinstance(layer, TorchCategoricalLayer):
            probs = list(layer.parameters())[0]

            # Compute current step
            ins, pl = self._log_likelihoods[layer]  # (B, K) or (B,)
            suff_stats = (
                torch.nn.functional.one_hot(ins.long()).squeeze(2).type(pl.type())
            )
            pl = pl.grad
            numerator = torch.einsum("ilj, ilk->ijk", pl, suff_stats).clamp(alpha)
            denominator = pl.sum(dim=1)

            if self.grad_acc:
                if layer in self._accumulator_categorical:
                    num, den = self._accumulator_categorical[layer]
                    numerator = numerator.detach() + num
                    denominator = denominator.detach() + den
                self._accumulator_categorical[layer] = (numerator, denominator)
                if self.grad_acc == self._current_acc_step:
                    exp_params = numerator / denominator.clamp(min=alpha).unsqueeze(-1)
                    probs.data.lerp_(exp_params, weight=lr)

            else:
                exp_params = numerator / denominator.clamp(min=alpha).unsqueeze(-1)
                probs.data.lerp_(exp_params, weight=lr)

        elif isinstance(layer, TorchGaussianLayer):
            ins, pl = self._log_likelihoods[layer]
            pl = pl.grad
            suff_stats = torch.stack((ins, ins**2), dim=-1)
            pl = pl.unsqueeze(-1)
            exp_params = torch.sum(suff_stats * pl, dim=1) / pl.sum(dim=1).clamp(
                min=alpha
            )
            mean, stdev = list(layer.parameters())
            new_mean = exp_params[..., 0].unsqueeze(-2)
            new_stddev = torch.sqrt(
                (exp_params[..., 1] - exp_params[..., 0] ** 2).clamp(min=alpha)
            ).unsqueeze(-2)
            mean.data.lerp_(new_mean, weight=lr)
            stdev.data.lerp_(new_stddev, weight=lr)

        elif isinstance(layer, TorchHadamardLayer):
            # no-op or custom handling if needed
            pass
        else:
            raise NotImplementedError(f"EM update not implemented for {type(layer)}")

    def step(self, closure: Optional[callable] = None):
        # closure not required; we follow pattern: compute log-probs, backward, then apply EM updates
        # if closure is None:
        #     raise ValueError(
        #         "This optimizer requires a closure that returns the batch tensor to evaluate."
        #     )
        # batch = closure()
        # self._log_likelihoods.clear()
        #
        # output_log_probs = self.pc.evaluate(batch, module_fn=self.em_forward)  # (B,)
        # closure().backward()

        closure()
        self._current_acc_step += 1
        # lightning closure automatically call backward on the loss
        for module in self.pc.topological_ordering():
            self.em_backward(module)

        self._log_likelihoods.clear()
        if self.grad_acc == 0 or self._current_acc_step == self.grad_acc:
            self.pc.zero_grad()
            self._current_acc_step = 0
        # return output_log_probs

    def zero_grad(self, set_to_none=True):
        # no-op because we call pc.zero_grad() after updates, but keep for compatibility
        # try:
        #     self.pc.zero_grad()
        # except Exception:
        #     pass
        pass


def em_step(
    pc: TorchCircuit,
    batch: Tensor,  # batch shape: (B, D)
    lr: float,  # step size for the EM update; between 0 and 1
    # circuit_log_prob: Callable, # function that computes the log probabilities of the circuit given the batch
    # contains_nans: bool, # whether the training or validation data contains NaNs
    n_chunks: int = 1,  # chunked computation to avoid OOM
    alpha: float = 1e-5,  # small constant to avoid zero probabilities
) -> Tensor:
    """Perform a single EM step on the given batch of data. Updates the parameters of the PC in-place.
    Args:
        pc: The probabilistic circuit to be trained.
        batch: Input data batch of shape (B, D).
        lr: Step size for the EM update; between 0 and 1.
        contains_nans: Whether the training or validation data contains NaNs.
        n_chunks: Number of chunks to split the batch into to avoid OOM.
        alpha: Small constant to avoid zero probabilities.
    Returns:
        Updated parameters of the PC after the EM step.
    """
    log_likelihoods = {}

    def em_forward(layer: TorchLayer, *inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(layer, TorchInputLayer):
            output = layer(inputs)  # TODO This assumes log-semiring
            output.retain_grad()

            log_likelihoods[layer] = (input, output)

        else:
            output = layer(*inputs)

        return output

    def circuit_log_prob(X: torch.Tensor) -> torch.Tensor:
        # Compute log probabilities for data X using the circuit
        return pc.evaluate(X, module_fn=em_forward)

    output_log_probs = circuit_log_prob(
        batch
    )  # shape: output_log_probs (B,), all_outputs: list of layer outputs

    output_log_probs.sum().backward()

    @torch.no_grad()
    def em_backward(layer: TorchLayer, *inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(layer, TorchSumLayer) or isinstance(layer, TorchCPTLayer):
            # TODO This assumes no parametrizations and a single parameter
            weight = list(layer.parameters())[0]
            old_weight = weight.clone()
            weight.mul_(weight.grad).clamp(alpha)
            weight.div_(weight.sum(dim=-1, keepdim=True))
            weight.data = old_weight.lerp(weight, weight=lr)
        elif isinstance(layer, TorchCategoricalLayer):
            probs = list(layer.parameters())[0]
            pl = log_likelihoods[layer].grad
            ins = inputs[0]
            suff_stats = torch.nn.functional.one_hot(ins.long())
            pl = pl.unsqueeze(-1)
            exp_params = torch.sum(suff_stats * pl, dim=1) / pl.sum(dim=1).clamp(
                min=alpha
            )

            new_probs = exp_params.unsqueeze(1)

            probs.data.lerp_(new_probs, weight=lr)

        elif isinstance(layer, TorchGaussianLayer):
            # TODO This assumes two parameters and no parametrization
            pl = log_likelihoods[layer].grad
            ins = inputs[0]

            suff_stats = torch.stack((ins, ins**2), dim=-1)
            pl = pl.unsqueeze(-1)
            exp_params = torch.sum(suff_stats * pl, dim=1) / pl.sum(dim=1).clamp(
                min=alpha
            )
            mean, stdev = list(layer.parameters())
            new_mean = exp_params[..., 0].unsqueeze(-2)
            new_stddev = torch.sqrt(
                (exp_params[..., 1] - exp_params[..., 0] ** 2).clamp(min=alpha)
            ).unsqueeze(-2)
            # TODO CHECK -> I don't know why there is in the example an extra dimension on the parameters (the documentation says there are 2, but I see 3)

            mean.data.lerp_(new_mean, weight=lr)
            stdev.data.lerp_(new_stddev, weight=lr)
        elif isinstance(layer, TorchHadamardLayer):
            pass
        else:
            raise NotImplementedError()

        return inputs[0]

    # Loop through layers and update their parameters with EM step
    pc.evaluate(batch, module_fn=em_backward)

    # Zero gradients after EM step
    pc.zero_grad()

    return output_log_probs
