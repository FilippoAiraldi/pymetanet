from collections.abc import Iterable
from functools import cached_property
from itertools import chain, product
from typing import Optional, Union

import networkx as nx

from sym_metanet.blocks.base import ElementBase, ElementWithVars
from sym_metanet.blocks.destinations import Destination
from sym_metanet.blocks.links import Link
from sym_metanet.blocks.nodes import Node
from sym_metanet.blocks.origins import MeteredOnRamp, Origin
from sym_metanet.engines.core import EngineBase
from sym_metanet.errors import InvalidNetworkError
from sym_metanet.util.funcs import invalidate_cache
from sym_metanet.util.types import VarType
from sym_metanet.views import (
    DESTINATIONENTRY,
    LINKENTRY,
    ORIGINENTRY,
    InLinkViewWrapper,
    OutLinkViewWrapper,
)


class Network(ElementBase):
    """Highway network."""

    def __init__(self, name: Optional[str] = None):
        """Instantiates an highway network.

        Parameters
        ----------
        name : str, optional
            Name of the network, by default `None`.
        """
        super().__init__(name)
        self._graph = nx.DiGraph(name=name)

    @property
    def G(self) -> nx.DiGraph:
        """Returns the underlying `networkx`'s graph of the highway."""
        return self._graph

    @property
    def graph(self) -> nx.DiGraph:
        """Returns the underlying `networkx`'s graph of the highway."""
        return self._graph

    @property
    def asgraph(self) -> nx.DiGraph:
        """Returns the underlying `networkx`'s graph of the highway."""
        return self._graph

    @property
    def nodes(self) -> nx.classes.reportviews.NodeView:
        """Returns a view on the nodes of the network."""
        return self._graph.nodes

    @cached_property
    def nodes_by_name(self) -> dict[str, Node]:
        return {node.name: node for node in self._graph.nodes}

    @cached_property
    def links(self) -> OutLinkViewWrapper:
        """Returns a view on the links of the network."""
        return OutLinkViewWrapper(self._graph)

    @property
    def out_links(self) -> OutLinkViewWrapper:
        """Alias of the `links` property."""
        return self.links

    @cached_property
    def in_links(self) -> InLinkViewWrapper:
        """Returns a view on the inward links of the network."""
        return InLinkViewWrapper(self._graph)

    @cached_property
    def links_by_name(self) -> dict[str, Link[VarType]]:
        return {  # type: ignore[var-annotated]
            link.name: link for _, _, link in self.links
        }

    @cached_property
    def nodes_by_link(self) -> dict[Link[VarType], tuple[Node, Node]]:
        return {  # type: ignore[var-annotated]
            link: (unode, dnode) for unode, dnode, link in self.links
        }

    @cached_property
    def origins(self) -> dict[Origin[VarType], Node]:
        return {
            data[ORIGINENTRY]: node
            for node, data in self._graph.nodes.data()
            if ORIGINENTRY in data
        }

    @cached_property
    def origins_by_name(self) -> dict[str, Origin[VarType]]:
        return {origin.name: origin for origin in self.origins}  # type: ignore[misc]

    @cached_property
    def origins_by_node(self) -> dict[Node, Origin[VarType]]:
        d = self.origins
        return dict(zip(d.values(), d.keys()))  # type: ignore[arg-type]

    @cached_property
    def destinations(self) -> dict[Destination[VarType], Node]:
        return {
            data[DESTINATIONENTRY]: node
            for node, data in self._graph.nodes.data()
            if DESTINATIONENTRY in data
        }

    @cached_property
    def destinations_by_name(self) -> dict[str, Destination[VarType]]:
        return {
            destination.name: destination  # type: ignore[misc]
            for destination in self.destinations
        }

    @cached_property
    def destinations_by_node(self) -> dict[Node, Destination]:
        d = self.destinations
        return dict(zip(d.values(), d.keys()))

    @property
    def elements(self) -> Iterable[ElementWithVars[VarType]]:
        """Gets an iterator to all the elements of the network."""
        return chain(
            (link[-1] for link in self.links),  # type: ignore[var-annotated]
            self.origins,  # type: ignore[arg-type]
            self.destinations,  # type: ignore[arg-type]
        )

    @property
    def states(self) -> dict[ElementWithVars[VarType], dict[str, VarType]]:
        """Gets the states of the network's elements."""
        return {
            el: el.states for el in self.elements if el.has_states  # type: ignore[misc]
        }

    @property
    def next_states(
        self,
    ) -> dict[ElementWithVars[VarType], dict[str, VarType]]:
        """Gets the states of the network's elements after stepping the
        dynamics for one time step."""
        return {
            el: el.next_states  # type: ignore[misc]
            for el in self.elements
            if el.has_next_states
        }

    @property
    def actions(self) -> dict[ElementWithVars[VarType], dict[str, VarType]]:
        """Gets the control action of the network's elements."""
        return {
            el: el.actions  # type: ignore[misc]
            for el in self.elements
            if el.has_actions
        }

    @property
    def disturbances(
        self,
    ) -> dict[ElementWithVars[VarType], dict[str, VarType]]:
        """Gets the disturbances of the network's elements."""
        return {
            el: el.disturbances  # type: ignore[misc]
            for el in self.elements
            if el.has_disturbances
        }

    @invalidate_cache(nodes_by_name)
    def add_node(self, node: Node) -> "Network":
        """Adds a node to the highway network.

        Parameters
        ----------
        node : Node
            Node to be added.

        Returns
        -------
        Network
            A reference to itself.
        """
        self._graph.add_node(node)
        return self

    @invalidate_cache(nodes_by_name)
    def add_nodes(self, nodes: Iterable[Node]) -> "Network":
        """Adds multiple nodes. See `Network.add_node`.

        Parameters
        ----------
        nodes : iterable of Nodes
            Nodes to be added.

        Returns
        -------
        Network
            A reference to itself.
        """
        self._graph.add_nodes_from(nodes)
        return self

    @invalidate_cache(links_by_name, nodes_by_link)
    def add_link(
        self, node_up: Node, link: Link[VarType], node_down: Node
    ) -> "Network":
        """Adds a link to the highway network, between two nodes.

        Parameters
        ----------
        node_up : Node
            Upstream node, that is, where traffic is coming from.
        link : Link
            The link to be added connecting the two nodes.
        node_down : Node
            Downstream node, that is, where traffic is going to.

        Returns
        -------
        Network
            A reference to itself.
        """
        self._graph.add_edge(node_up, node_down, **{LINKENTRY: link})
        return self

    @invalidate_cache(links_by_name, nodes_by_link)
    def add_links(self, links: Iterable[tuple[Node, Link[VarType], Node]]) -> "Network":
        """Adds multiple links. See `Network.add_link`.

        Parameters
        ----------
        nodes : iterable of Tuple[Node, Link, Node]
            Links to be added between the corresponding nodes.

        Returns
        -------
        Network
            A reference to itself.
        """

        def get_edge(linkdata: tuple[Node, Link[VarType], Node]):
            node_up, link, node_down = linkdata
            return (node_up, node_down, {LINKENTRY: link})

        self._graph.add_edges_from(get_edge(link) for link in links)
        return self

    @invalidate_cache(origins, origins_by_node, origins_by_name)
    def add_origin(self, origin: Origin[VarType], node: Node) -> "Network":
        """Adds the given traffic origin to the node.

        Parameters
        ----------
        origin : Origin
            Origin to be added to the network.
        node : Node
            Node which the origin is attached to.

        Returns
        -------
        Network
            A reference to itself.
        """
        if node not in self.nodes:
            self._graph.add_node(node, **{ORIGINENTRY: origin})
        else:
            self.nodes[node][ORIGINENTRY] = origin
        return self

    @invalidate_cache(destinations, destinations_by_node, destinations_by_name)
    def add_destination(
        self, destination: Destination[VarType], node: Node
    ) -> "Network":
        """Adds the given traffic destination to the node.

        Parameters
        ----------
        destination : Destination
            Destination to be added to the network.
        node : Node
            Node which the destination is attached to.

        Returns
        -------
        Network
            A reference to itself.
        """
        if node not in self.nodes:
            self._graph.add_node(node, **{DESTINATIONENTRY: destination})
        else:
            self.nodes[node][DESTINATIONENTRY] = destination
        self.destinations_by_name[
            destination.name
        ] = destination  # type: ignore[assignment]
        return self

    def add_path(
        self,
        path: Iterable[Union[Node, Link[VarType]]],
        origin: Optional[Origin[VarType]] = None,
        destination: Optional[Destination[VarType]] = None,
    ) -> "Network":
        """Adds a path of nodes and links between the origin and the
        destination.

        Parameters
        ----------
        path : iterable of either Nodes and Links
            A path consists of an alternating sequence of nodes and links,
            starting from the first node and ending at the last. For example, a
            valid path is: `node1, link1, node2, link2, node3, ..., nodeN`.
        origin : Origin, optional
            The origin where the path starts from. Pass `None` to have no
            origin attached to the first node in `path`.
        destination : Destination, optional
            The destination where the path ends in. Pass `None` to have no
            destination attached to the last node in `path`.

        Returns
        -------
        Network
            A reference to itself.

        Raises
        ------
        TypeError
            Raises if
            - the first or last points in `path` are not a `Node`
            - the alternation of `Link`s and `Node`s is not respected
            - the path has length 1, which is not accepted.
        """
        path = iter(path)
        first_node = next(path)
        if not isinstance(first_node, Node):
            raise TypeError(
                f"First element of the path must be a `{Node.__name__}`; got "
                f"{type(first_node)} instead."
            )
        self.add_node(first_node)
        if origin is not None:
            self.add_origin(origin, first_node)
        current_link: list[Union[Node, Link[VarType]]] = [first_node]

        longer_than_one = False
        for i, point in enumerate(path):
            longer_than_one = True
            current_link.append(point)
            L = len(current_link)
            if L == 2:
                if not isinstance(point, Link):
                    raise TypeError(
                        f"Expected a `{Link.__name__}` at index {i} of the path; "
                        f"got {type(point)} instead."
                    )
            else:  # L == 3
                if not isinstance(point, Node):
                    raise TypeError(
                        f"Expected a `{Node.__name__}` at index {i} of the path; "
                        f"got {type(point)} instead."
                    )
                self.add_node(point)
                self.add_link(*current_link)
                current_link = current_link[-1:]
        if not longer_than_one:
            raise ValueError("Path must be longer than a single node.")

        last_node = point
        if not isinstance(first_node, Node):
            raise TypeError(
                f"Last element of the path must be a `{Node.__name__}`; got "
                f"{type(first_node)} instead."
            )
        if destination is not None:
            self.add_destination(destination, last_node)
        return self

    def is_valid(self, raises: bool = False) -> tuple[bool, list[str]]:
        """Checks whether the network is consistent.

        Parameters
        ----------
        raises : bool, optional
            If `True`, if an issue is found, an exception is raised. If
            `False`, then the exception messages are returned in a list. By
            default, `False`.

        Returns
        -------
        bool
            `True` if the network is valid, `False` otherwise.
        list[str]
            A list of messages describing the invalid issues found. Returned
            only if `raises=False`.

        Raises
        ------
        InvalidNetworkError
            Raises if `raises=True` and if
             1) a link, origin or destination is duplicated in the network

             2) a node has both an origin and a destination

             3) a node has no entering and no exiting links

             4) a node with no entering links has no origin

             5) a node with no exiting links has no destination

             6) a node with an origin (not a ramp) has also entering links

             7) a node with an origin has multiple exiting links

             8) a node with a destination has multiple entering links

             9) a node with a destination has also exiting links.
        """
        msgs = []

        def origin_destination_yielder():
            for data, entry in product(
                self._graph.nodes.values(), (ORIGINENTRY, DESTINATIONENTRY)
            ):
                if entry in data:
                    yield data[entry]

        # (1)
        count: dict[object, int] = {}
        for o in chain(
            (link[2] for link in self.links), iter(origin_destination_yielder())
        ):
            d = count.get(o, 0) + 1
            if d > 1:
                msgs.append(f"Element {o.name} is duplicated in the network.")
                if raises:
                    raise InvalidNetworkError(msgs[-1])
            count[o] = d

        # (2); (3); (4); (5)
        for node, nodedata in self.nodes.data():
            if ORIGINENTRY in nodedata and DESTINATIONENTRY in nodedata:
                msgs.append(
                    f"Node {node.name} must either have an origin or a "
                    "destination, but not both."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])
            n_in, n_out = len(self.in_links(node)), len(self.out_links(node))
            if n_in == 0 and n_out == 0:
                msgs.append(f"Node {node.name} is connected to no link.")
                if raises:
                    raise InvalidNetworkError(msgs[-1])
            if n_in == 0 and ORIGINENTRY not in nodedata:
                msgs.append(
                    f"Node {node.name} has neither any entering links nor an origin."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])
            if n_out == 0 and DESTINATIONENTRY not in nodedata:
                msgs.append(
                    f"Node {node.name} has neither any exiting links nor a destination."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])

        # (6); (7)
        for origin, node in self.origins.items():
            if not isinstance(origin, MeteredOnRamp) and any(self.in_links(node)):
                msgs.append(
                    f"Expected node {node.name} to have no entering links, as "
                    f"it is connected to origin {origin.name} (only ramps "
                    "support entering links)."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])
            if len(self.out_links(node)) > 1:
                msgs.append(
                    f"Expected node {node.name} to have at most one exiting "
                    f"link, as it is connected to origin {origin.name}."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])

        # (8); (9)
        for destination, node in self.destinations.items():
            if len(self.in_links(node)) > 1:
                msgs.append(
                    f"Expected node {node.name} to have at most one entering "
                    "link, as it is connected to destination " + destination.name + "."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])
            if any(self.out_links(node)):
                msgs.append(
                    f"Expected node {node.name} to have no exiting links, as "
                    f"it is connected to destination {destination.name}."
                )
                if raises:
                    raise InvalidNetworkError(msgs[-1])
        return not msgs, msgs

    def step(
        self,
        init_conditions: Optional[
            dict[ElementWithVars[VarType], dict[str, VarType]]
        ] = None,
        engine: Optional[EngineBase] = None,
        positive_init_speed: bool = False,
        positive_init_density: bool = False,
        positive_init_queue: bool = False,
        positive_next_speed: bool = False,
        positive_next_density: bool = False,
        positive_next_queue: bool = False,
        **other_parameters: VarType,
    ) -> None:
        """Steps the dynamics of the network's elements.

        Parameters
        ----------
        init_conditions : dict[element, dict[varname, variable]], optional
            For each element in the network, provides name-variable tuples to
            initialize states, actions and disturbances with specific values.
            These values must be compatible with the symbolic engine in type
            and shape. If not provided, variables are initialized
            automatically.
        engine : EngineBase, optional
            The engine to be used for stepping the dynamics. If `None`, the
            current engine is used.
        positive_init_speed, positive_init_density, positive_init_queue: bool, optional
            If `True`, forces the initial speeds/densities/queues to be positive, e.g.,
            as `v = max(0, v)`. METANET is in fact known to sometime yield negative
            quantities, which are infeasible in reality.
        positive_next_speed, positive_next_density, positive_next_queue: bool, optional
            Similarly to previous, but for the next values of the variables, i.e., at
            the next time step.
        other_parameters : variables
            All the other parameters (e.g., sampling time) required during
            the computations.
        """
        # initialization
        if init_conditions is None:
            init_conditions = {}
        for el in self.elements:
            el.init_vars(
                init_conditions=init_conditions.get(el),  # type: ignore[arg-type]
                engine=engine,
                positive_init_queue=positive_init_queue,
                positive_init_density=positive_init_density,
                positive_init_speed=positive_init_speed,
            )

        # dynamics
        for origin in self.origins:
            origin.step(
                net=self,
                engine=engine,
                positive_next_queue=positive_next_queue,
                **other_parameters,
            )
        for _, _, link in self.links:  # type: ignore[var-annotated]
            link.step(
                net=self,
                engine=engine,
                positive_next_speed=positive_next_speed,
                positive_next_density=positive_next_density,
                **other_parameters,
            )
