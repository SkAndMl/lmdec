from dataclasses import dataclass


@dataclass(frozen=True)
class DenseMLP:
    intermediate_size: int
    gated: bool = True

    def parameters(self, hidden_size: int, num_layers: int) -> int:
        return num_layers * self.parameters_per_layer(hidden_size)

    def parameters_per_layer(self, hidden_size: int) -> int:
        matrices = 3 if self.gated else 2
        return matrices * hidden_size * self.intermediate_size


@dataclass(frozen=True)
class MixtureOfExperts:
    dense: DenseMLP
    num_dense_layers: int
    expert_intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    num_activated_experts: int

    def __post_init__(self) -> None:
        if self.num_dense_layers < 0:
            raise ValueError(
                f"num_dense_layers ({self.num_dense_layers}) cannot be negative"
            )
        if self.num_activated_experts > self.num_routed_experts:
            raise ValueError(
                f"num_activated_experts ({self.num_activated_experts}) cannot exceed "
                f"num_routed_experts ({self.num_routed_experts})"
            )

    def parameters(self, hidden_size: int, num_layers: int) -> int:
        if self.num_dense_layers > num_layers:
            raise ValueError(
                f"num_dense_layers ({self.num_dense_layers}) cannot exceed "
                f"num_layers ({num_layers})"
            )

        moe_layers = num_layers - self.num_dense_layers

        return self.dense.parameters(
            hidden_size, self.num_dense_layers
        ) + moe_layers * self.parameters_per_moe_layer(hidden_size)

    def parameters_per_moe_layer(self, hidden_size: int) -> int:
        expert = 3 * hidden_size * self.expert_intermediate_size
        router = self.num_routed_experts * hidden_size
        experts = self.num_routed_experts + self.num_shared_experts

        return router + experts * expert
