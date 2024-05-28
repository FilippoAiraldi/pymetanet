from collections.abc import Collection
from typing import TYPE_CHECKING, Optional

from sym_metanet.blocks.base import ElementBase
from sym_metanet.engines.core import EngineBase, get_current_engine
from sym_metanet.util.funcs import first
from sym_metanet.util.types import Variable, VarType

if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link
    from sym_metanet.network import Network


class Node(ElementBase):
    """Node of the highway [1, Section 3.2.2] representing, e.g., the connection between
    two links. Nodes do not correspond to actual physical components of the highway, but
    are used to separate links in case there is a major change in the link parameters or
    a junction or bifurcation.

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    def get_downstream_density(
        self, net: "Network", engine: Optional[EngineBase] = None, **kwargs
    ) -> Variable:
        """Computes the (virtual) downstream density of the node.

        Parameters
        ----------
        net : Network
            The network which node and link belongs to.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        symbolic variable
            Returns the (virtual) downstream density.
        """
        if engine is None:
            engine = get_current_engine()

        # following the link entering this node, this node can only be a
        # destination or have multiple exiting links
        if self in net.destinations_by_node:
            return net.destinations_by_node[self].get_density(
                net, engine=engine, **kwargs
            )

        # if no destination, then there must be 1 or more exiting links
        links_down: Collection[tuple["Node", "Node", "Link[Variable]"]] = net.out_links(
            self
        )
        if len(links_down) == 1:
            return first(links_down)[-1].states["rho"][0]
        rho_firsts = engine.vcat(
            *(dlink.states["rho"][0] for _, _, dlink in links_down)
        )
        return engine.nodes.get_downstream_density(rho_firsts)

    def get_upstream_speed_and_flow(
        self,
        net: "Network",
        link: "Link[VarType]",
        engine: Optional[EngineBase] = None,
        **kwargs,
    ) -> tuple[VarType, VarType]:
        """Computes the (virtual) upstream speed and flow of the node for the current
        link.

        Parameters
        ----------
        net : Network
            The network which node and link belongs to.
        link : Link
            The current link (which departs from this node) querying this information
            from the node.
        engine : EngineBase, optional
            The engine to be used. If `None`, the current engine is used.

        Returns
        -------
        tuple[symbolic variable, symbolic variable]
            Returns the (virtual) upstream speed and flow.
        """
        if engine is None:
            engine = get_current_engine()

        # the node can have 1 or more entering links, as well as a ramp origin.
        # Speed is dictated by the entering links, if any; otherwise by the
        # origin (same as first segment). Flow is dictated both by entering
        # links and origin.
        links_up: Collection[tuple["Node", "Node", "Link[VarType]"]] = net.in_links(
            self
        )
        n_up = len(links_up)
        if self in net.origins_by_node:
            origin = net.origins_by_node[self]  # type: ignore[index]
            v_o = origin.get_speed(net, engine=engine, **kwargs)
            q_o = origin.get_flow(net, engine=engine, **kwargs)
        else:
            v_o = None
            q_o = None

        if n_up == 0:
            v = v_o
            q = q_o
        elif n_up == 1:
            link_up = next(iter(links_up))[-1]
            v = link_up.states["v"][-1]
            q = link_up.get_flow(engine)[-1]
            if q_o is not None:
                q += q_o  # type: ignore[assignment,operator]
        else:
            v_last = []
            q_last = []
            for _, _, link_up in links_up:
                v_last.append(link_up.states["v"][-1])
                q_last.append(link_up.get_flow(engine)[-1])
            v_last = engine.vcat(*v_last)
            q_last = engine.vcat(*q_last)
            links_down: Collection[
                tuple["Node", "Node", "Link[VarType]"]
            ] = net.out_links(self)
            betas = engine.vcat(*(dlink.turnrate for _, _, dlink in links_down))

            v = engine.nodes.get_upstream_speed(q_last, v_last)
            q = engine.nodes.get_upstream_flow(q_last, link.turnrate, betas, q_o)
        return v, q  # type: ignore[return-value]
