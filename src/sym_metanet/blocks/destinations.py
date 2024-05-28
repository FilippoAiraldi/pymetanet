from typing import TYPE_CHECKING, Optional, Union

from sym_metanet.blocks.base import ElementWithVars
from sym_metanet.engines.core import EngineBase, get_current_engine
from sym_metanet.util.funcs import first
from sym_metanet.util.types import VarType

if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link
    from sym_metanet.network import Network


class Destination(ElementWithVars[VarType]):
    """Ideal congestion-free destination, representing a sink where cars can leave the
    highway with no congestion (i.e., no slowing down due to downstream density)."""

    def init_vars(self, *_, **__) -> None:
        """Initializes no variable in the ideal destination."""

    def step_dynamics(self, *_, **__) -> dict[str, VarType]:
        """No dynamics to steps in the ideal destination."""
        return {}

    def get_density(
        self, net: "Network", engine: Optional[EngineBase] = None, **_
    ) -> VarType:
        """Computes the (downstream) density induced by the ideal destination.

        Parameters
        ----------
        net : Network
            The network this destination belongs to.

        Returns
        -------
        symbolic variable
            The destination's downstream density.
        """
        if engine is None:
            engine = get_current_engine()
        link_up = self._get_entering_link(net)
        return engine.destinations.get_congestion_free_downstream_density(
            link_up.states["rho"][-1], link_up.rho_crit
        )

    def _get_entering_link(self, net: "Network") -> "Link[VarType]":
        """Internal utility to fetch the link entering this destination (can only be
        one)."""
        links_up = net.in_links(net.destinations[self])
        assert (
            len(links_up) == 1
        ), "Internal error. Only one link can enter a destination."
        return first(links_up)[-1]


class CongestedDestination(Destination[VarType]):
    """Destination with a downstream density scenario to emulate congestions, that is,
    cars cannot exit freely the highway but must slow down and, possibly, create a
    congestion."""

    _disturbances = {"d"}

    def init_vars(
        self,
        init_conditions: Optional[dict[str, VarType]] = None,
        engine: Optional[EngineBase] = None,
        **_,
    ) -> None:
        """Initializes
        - `d`: downstream density scenario (disturbance).

        Parameters
        ----------
        net : Network
            The network this destination belongs to.
        init_conditions : dict[str, variable], optional
            Provides name-variable tuples to initialize states, actions and disturbances
            with specific values. These values must be compatible with the symbolic
            engine in type and shape. If not provided, variables are initialized
            automatically.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.
        """
        if engine is None:
            engine = get_current_engine()
        self.disturbances: dict[str, VarType] = {
            "d": engine.var(f"d_{self.name}")
            if init_conditions is None or "d" not in init_conditions
            else init_conditions["d"]
        }

    def get_density(
        self, net: "Network", engine: Optional[EngineBase] = None, **_
    ) -> VarType:
        """Computes the (downstream) density induced by the congested destination.

        Parameters
        ----------
        net : Network
            The network this destination belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        variable
            The destination's downstream density.
        """
        if engine is None:
            engine = get_current_engine()
        link_up = self._get_entering_link(net)
        return engine.destinations.get_congested_downstream_density(
            link_up.states["rho"][-1], self.disturbances["d"], link_up.rho_crit
        )


class OffRamp(Destination[VarType]):
    """Unmetered off-ramp destination. Incoming vehicles can leave the highway via this
    off-ramp or continue via the downstream links, according to the turn rate of these
    elements.

    Note: if this destination is attached to a node with no other exiting links, it will
    act as an uncongested destination `Destination`."""

    def __init__(
        self,
        turnrate: Union[VarType, float] = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """Initializes no variable in the ideal destination."""
        super().__init__(name)
        self.turnrate = turnrate

    def get_flow(
        self,
        net: "Network",
        engine: Optional[EngineBase] = None,
        q_up: Optional[VarType] = None,
        q_orig: Optional[VarType] = None,
        betas_down: Optional[VarType] = None,
        **kwargs,
    ) -> VarType:
        """Computes the flow exiting the highway via the off-ramp.

        Parameters
        ----------
        net : Network
            The network this off-ramp belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.
        q_up : var type, optional
            Flows of the entering link, by default `None`, in which case it is computed
            automatically.
        q_orig : var type, optional
            Flows of the origin (if any), by default `None`, in which case it is
            computed automatically.
        betas_down : var type, optional
            Turn rates of the downstream links, by default `None`, in which case it is
            computed automatically.

        Returns
        -------
        var type
            The flow exiting the highway via the off-ramp.
        """
        if engine is None:
            engine = get_current_engine()

        # if not passed, get the upstream flow from the entering link and origin, if any
        node = net.destinations[self]
        if q_up is None:
            link_up = self._get_entering_link(net)
            q_up = link_up.get_flow(engine, **kwargs)[-1]
        if q_orig is None and node in net.origins_by_node:
            origin = net.origins_by_node[node]
            q_orig = origin.get_flow(net, engine=engine, **kwargs)

        # if not passed, compute the fraction of upstream flow exiting the highway via
        # this off-ramp
        if betas_down is None:
            links_down = net.out_links(node)
            betas = engine.vcat(*(l.turnrate for _, _, l in links_down), self.turnrate)
        else:
            betas = engine.vcat(betas_down, self.turnrate)
        return engine.nodes.get_upstream_flow(q_up, self.turnrate, betas, q_orig)
