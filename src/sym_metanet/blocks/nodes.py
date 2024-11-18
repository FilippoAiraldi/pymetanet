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
        # the node can have 0 or more exiting links, as well as a destination. If the
        # destination is final (i.e., there are no exiting links), then the downstream
        # density is dictated only by the destination. Otherwise, the destination must
        # be an off-ramp, which is regarded as an additional link, and the downstream
        # density is computed via the densities of the exiting links and the off-ramp.
        if engine is None:
            engine = get_current_engine()

        # first, grab the density of the destination, if any
        density_dest = (
            net.destinations_by_node[self].get_density(net, engine=engine, **kwargs)
            if self in net.destinations_by_node
            else None
        )

        # then, process the exiting links
        links_down = net.out_links(self)
        n_down = len(links_down)
        if n_down == 0:
            # no exiting link, only a destination (which must be final)
            assert density_dest is not None, "No exiting links or destination!"
            return density_dest
        elif n_down == 1 and density_dest is None:
            # if only one exiting link and no destination, simply return link's density
            return first(links_down)[-1].states["rho"][0]
        else:
            # 1 or more exiting links and a destination, so combine their densities
            rho_firsts = [dlink.states["rho"][0] for _, _, dlink in links_down]
            if density_dest is not None:
                rho_firsts.append(density_dest)
            return engine.nodes.get_downstream_density(engine.vcat(*rho_firsts))

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
        # the node can have 0 or more entering links, as well as a ramp origin and a
        # destination. Speed is dictated by the entering links, if any; otherwise by the
        # origin (same as first segment). Flow is dictated both by entering
        # links, origin, and destination.
        if engine is None:
            engine = get_current_engine()

        # first of all, grab speed and flow of entering links and origin, if any - in a
        # valid network, every node must have at least an entering link or an origin
        links_up = net.in_links(self)
        origin = net.origins_by_node.get(self, None)
        assert links_up or origin is not None, "No entering links or origin!"
        v_up = []
        q_up = []
        for _, _, link_up in links_up:
            v_up.append(link_up.states["v"][-1])
            q_up.append(link_up.get_flow(engine)[-1])
        if origin is not None:
            v_o = origin.get_speed(net, engine=engine, **kwargs)
            q_o = origin.get_flow(net, engine=engine, **kwargs)
        else:
            v_o = q_o = 0

        # then, compute the upstream speed and adjust q_up accordingly
        n_up = len(links_up)
        if n_up == 0:
            v = v_o
            q_up, q_o = q_o, 0  # swap origin's flow with link's, and set origin's to 0
        elif n_up == 1:
            v = v_up[0]
            q_up = q_up[0]
        else:
            q_up = engine.vcat(*q_up)
            v_up = engine.vcat(*v_up)
            v = engine.nodes.get_upstream_speed(q_up, v_up)

        # finally, compute the upstream flow - in a valid network, every node must have
        # at least an exiting link or a destination, or both iff the destination is an
        # off-ramp (i.e., it has a `get_flow` method)
        links_down = net.out_links(self)
        assert any(
            link is link_down for _, _, link_down in links_down
        ), "Link not contained in entering links!"
        betas = engine.vcat(*(link_down.turnrate for _, _, link_down in links_down))
        if self in net.destinations_by_node:
            destination = net.destinations_by_node[self]
            assert hasattr(destination, "get_flow"), "Destination is not an off-ramp!"
            q_d = destination.get_flow(
                net,
                engine=engine,
                q_up=q_up,
                q_orig=q_o,
                betas_down=betas,
                **kwargs,
            )
        else:
            q_d = None
        q = engine.nodes.get_upstream_flow(q_up, link.turnrate, betas, q_o, q_d)
        return v, q
