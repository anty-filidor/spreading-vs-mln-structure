"""`torch`-based Multilayer Independent Cascade Model."""

from typing import Any

import torch
import network_diffusion as nd


class TorchMICModel:
    """Multilayer Independent Cascade Model implemented in PyTorch."""

    def __init__(self, protocol: str, probability: float) -> None:
        """
        Create the object.

        :param protocol: logical operator that determines how to activate actor can be OR (then 
            actor gets activated if it gets positive input in one layer) or AND (then actor gets 
            activated if it gets positive input in all layers)
        :param probability: threshold parameter which activate actor (a random variable must be 
            smaller than this param to result in activation)
        """
        assert 0 <= probability <= 1, f"incorrect probability: {probability}!"
        self.probability = probability
        if protocol == "AND":
            self.protocol = self.protocol_AND
        elif protocol == "OR":
            self.protocol = self.protocol_OR
        else:
            raise ValueError("Only AND & OR value are allowed!")

    @staticmethod
    def protocol_AND(S_raw: torch.Tensor, net: nd.MultilayerNetworkTorch) -> torch.Tensor:
        """
        Aggregate positive impulses from the layers using AND strategy.

        :param S_raw: raw impulses obtained by the nodes
        :param net: a network which is a medium for the diffusion
        :return: a tensor shaped as [1 x number of actors] with 1. denoting activated actors in this
            simulation step and 0. denoting actors that weren't activated
        """
        return (S_raw + net.nodes_mask > 0).all(dim=0).to(torch.float)

    @staticmethod
    def protocol_OR(S_raw: torch.Tensor, net: nd.MultilayerNetworkTorch) -> torch.Tensor:
        """
        Aggregate positive impulses from the layers using OR strategy.

        :param S_raw: raw impulses obtained by the nodes
        :param net: a network which is a medium for the diffusion
        :return: a tensor shaped as [1 x number of actors] with 1. denoting activated actors in this
            simulation step and 0. denoting actors that weren't activated
        """
        return (S_raw > 0).any(dim=0).to(torch.float)

    @staticmethod
    def draw_live_edges(A: torch.Tensor, p: float) -> torch.Tensor:
        """
        Draw eges which transmit the state (i.e. their random weight < p).

        :param A: adjacency matrix as a sparse tensor shaped as `[nb layers x nb nodes x nb nodes]`
        :param p: threshold parameter which activate actor (a random variable must be smaller than
            this param to result in activation)
        :return: a filtered sparse adjacency matrix with edges that drawn numbers < p
        """
        raw_signals = torch.rand_like(A.values(), dtype=float)
        thre_signals = (raw_signals < p).to(float)
        T = torch.sparse_coo_tensor(indices=A.indices(), values=thre_signals, size=A.shape)
        # assert A.shape == T.shape, f"{A.shape} != {T.shape}"
        # assert ((A - T).to_dense() < 0).sum() == 0
        return T

    @staticmethod
    def mask_S_from(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a dense mask for T which discards signals from nodes which state != 1."""
        return (S > 0).to(torch.int).unsqueeze(-1)

    @staticmethod
    def mask_S_to(S: torch.Tensor) -> torch.Tensor:
        """Create a dense mask for T which discards signals to nodes which state != 0."""
        return torch.abs(torch.abs(S) - 1)

    def get_active_nodes(self, T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Obtain newly active nodes (0 -> 1) in the current simulation step.

        :param T: a filtered sparse adjacency matrix with edges that drawn numbers < p.
        :param S: a dense tensor of nodes' states (0 - inactive, 1 - active, -1 - activated,
            -inf - node does not exist).
        :return: a dense tensor shaped as S valued by 0s and 1s for newly activated nodes. 
        """
        S_f = self.mask_S_from(S)
        S_t = self.mask_S_to(S)
        S_new = ((T * S_f).sum(dim=1) * S_t).to_dense()
        return S_new

    @staticmethod
    def decay_active_nodes(S: torch.Tensor) -> torch.Tensor:
        """
        Change states of nodes that are active to become activated (aka removed) - (1 -> -1).

        :param S: a tensor of nodes' states (0 - inactive, 1 - active, -1 - activated, -inf - node
            does not exist).
        """
        decayed_S = -1. * torch.abs(S)
        decayed_S[decayed_S == -0.] = 0.
        return decayed_S

    def simulation_step(self, net: nd.MultilayerNetworkTorch, S0: torch.Tensor) -> torch.Tensor:
        """
        Perform a single simulation step.
        
        1. determine which edges drawn value below p
        2. transfer state from active (1.) nodes to their inactive (0.) neighbours only if egdes were preserved at step 1.
        3. aggregate positive impulses from the layers to determine actors that got activated during this simulation step 
        4. decay activation potential for actors that were acting as the active in the current simulation step
        5. obtain the final tensor of states after this simulation step 

        :param net: a network wtihch is a medium of the diffusion
        :param p: a probability of activation between active and inactive node
        :param protocol: a function that aggregates positive impulses from the network's layers
        :param S0: initial tensor of nodes' states (0 - inactive, 1 - active, -1 - activated, -inf - node does not exist)
        :return: updated tensor with nodes' states
        """
        T = self.draw_live_edges(net.adjacency_tensor, self.probability)
        S1_raw = self.get_active_nodes(T, S0)
        S1_aggregated = self.protocol(S_raw=S1_raw, net=net)
        S0_decayed = self.decay_active_nodes(S0)
        return S1_aggregated + S0_decayed


class TorchMICSimulator:
    """Simulator for TorchMICModel."""

    def __init__(
        self,
        model: TorchMICModel,
        net: nd.MultilayerNetworkTorch,
        n_steps: int,
        seed_set: set[Any],
        device: str | torch.device,
        debug: bool = False
    ) -> None:
        """
        Create the object.

        :param network:
        :param model:
        """
        self.model = model
        self.net = net
        self.n_steps = n_steps
        self.seed_set = seed_set
        self.debug = debug
        self.device = device
        self.validate_device(device)
        net.device = self.device
    
    @staticmethod
    def validate_device(device: str | torch.device) -> None:
        """
        Validate if the given device is available in PyTorch.

        :param device: The device to validate, e.g., "cpu", "cuda", "cuda:0".
        :raises ValueError: If the device is not available.
        """
        try:
            torch.device(device)
        except RuntimeError as e:
            raise ValueError(f"Invalid device specification: {device}") from e

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available on this machine!")

        if str(device).startswith("cuda:"):
            device_idx = int(str(device).split(":")[1])
            device_idx_max = torch.cuda.device_count() - 1
            if device_idx > device_idx_max:
                raise ValueError(f"Device index '{device_idx}' out of range [0; {device_idx_max}]!")

    def create_states_tensor(self, net: nd.MultilayerNetworkTorch, seed_set: set[Any]) -> torch.Tensor:
        """
        Create tensor of states

        :param net: a network (in tensor representation) to create a states tensor for
        :param seed_set: a set of initially active actors (ids of actors given in the original form)
        :return: a tensor shaped as [number_of_layers x number_of_actors] with 1. marked for seed nodes
            and -inf for nodes that were artifically added during converting the network to the tensor
            representation
        """
        seed_set_mapped = [net.actors_map[seed] for seed in seed_set]
        # if self.debug: print(f"{seed_set} -> {seed_set_mapped}")
        states_raw = torch.clone(net.nodes_mask)
        states_raw[states_raw == 1.] = -1 * float("inf")
        states_raw[:, seed_set_mapped] += 1
        return states_raw

    @staticmethod
    def is_steady_state(S_i: torch.Tensor, S_j: torch.Tensor) -> bool:
        """Check if consecutive states' tensors equal (i.e. simulation reached a steady state)."""
        if torch.equal(S_i, S_j):
            return True
        return False

    @staticmethod
    def S_nodes_to_actors(S: torch.Tensor) -> torch.Tensor:
        """Convert tensor of nodes' states to a vector of actors' states."""
        _S = torch.clone(S)
        _S[_S == -1 * float("inf")] = 0.
        return _S.sum(dim=0).clamp(-1, 1)

    def count_states(self, S: torch.Tensor) -> dict[int, int]:
        """Count actors not_exposed (0), exposed (-1) and active (1)."""
        states_values, states_counts = self.S_nodes_to_actors(S).unique(return_counts=True)
        return {int(val.item()): int(cnt.item()) for val, cnt in zip(states_values, states_counts)}

    def perform_propagation(self) -> dict[str, Any]:
        """Perform propagation and return a dictionary with global results."""
        simulation_length = None
        exposed = None
        not_exposed = None
        peak_infected = 1
        peak_iteration = 0
        expositions_rec = [len(self.seed_set)]

        S_i = self.create_states_tensor(self.net, self.seed_set)
        # if self.debug: print(f"Step: 0, actor-wise states: {self.count_states(S_i)}")

        for j in range(1, self.n_steps):

            S_j = self.model.simulation_step(self.net, S_i)
            step_result = self.count_states(S_j)
            # if self.debug: print(f"Step: {j}, actor-wise states: {step_result}")

            if step_result.get(1, 0) > peak_infected:
                peak_infected = step_result.get(1, 0)
                peak_iteration = j

            expositions_rec.append(step_result.get(1, 0))  # it's necessary to have this line here
            
            if self.is_steady_state(S_i, S_j) or j == self.n_steps - 1:
                # if self.debug: print(f"Simulation stopped after {j}th step")
                simulation_length = j + 1
                exposed = step_result.get(-1, 0) + step_result.get(1, 0)
                not_exposed = step_result.get(0, 0)
                break

            S_i = S_j

        return {
            "simulation_length": simulation_length,
            "exposed": exposed,
            "not_exposed": not_exposed,
            "peak_infected": peak_infected,
            "peak_iteration": peak_iteration,
            "expositions_rec": expositions_rec,
        }
