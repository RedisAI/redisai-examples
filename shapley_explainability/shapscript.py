from typing import Optional

import torch
from torch import Tensor, nn


class ShapleyValueSampling(nn.Module):
    """
    TorchScript-able implementation of Shapley Value Sampling.
    See https://captum.ai/api/shapley_value_sampling.html for
    reference. From that source:

    A perturbation based approach to compute attribution, based on the concept
    of Shapley Values from cooperative game theory. This method involves taking
    a random permutation of the input features and adding them one-by-one to the
    given baseline. The output difference after adding each feature corresponds
    to its attribution, and these difference are averaged when repeating this
    process n_samples times, each time choosing a new random permutation of
    the input features.
    """

    def __init__(
        self,
        model,
        n_samples: int = 20,
        baselines: Optional[Tensor] = None,
        target: Optional[int] = None,
    ):
        """
        Args:
            model: nn.Module to test, it is used as a Callable; it is assumed
                    that it is TorchScript-able for the current module to be
                    exported to TorchScript
            n_samples: number of random feature permutations performed
            baselines: reference values which replace each feature when
                    ablated; if no baselines are provided, baselines are set
                    to all zeros
            target: output indices for which Shapley Value Sampling is
                    computed; if model returns a single scalar, target can be
                    None
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.baselines = baselines
        self.target = target

    def generate_permutations(self, x):
        n_features = torch.numel(x[0])
        return [torch.randperm(n_features) for _ in range(self.n_samples)]

    def index_with_target(self, x):
        x_t = torch.transpose(x, 0, -1)
        x_target_t = x_t[self.target]
        return torch.transpose(x_target_t, 0, -1)

    def forward(self, x: Tensor) -> Tensor:
        attrib = torch.zeros_like(x, dtype=torch.float32)

        if self.baselines is not None:
            baselines = self.baselines
        else:
            baselines = torch.zeros_like(x)

        permutations = self.generate_permutations(x)

        n_features = torch.numel(x[0])

        for permutation in permutations:
            current = x.clone()
            for batch_i in range(current.shape[0]):
                current[batch_i] = baselines[
                    int(torch.randint(low=0, high=baselines.shape[0], size=(1,)))
                ]
            prev_out = self.model(current)
            prev_out_target = (
                self.index_with_target(prev_out)
                if self.target is not None
                else prev_out
            )

            for feature_i in range(n_features):
                permuted_feature_i = int(permutation[feature_i])
                current[:, permuted_feature_i] = x[:, permuted_feature_i]
                out = self.model(current)
                out_target = (
                    self.index_with_target(out) if self.target is not None else out
                )
                attrib[:, permuted_feature_i] += out_target - prev_out_target
                prev_out_target = out_target

        attrib /= self.n_samples

        return attrib
