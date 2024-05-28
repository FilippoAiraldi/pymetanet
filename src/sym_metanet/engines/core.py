from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Literal, Union

import sym_metanet
from sym_metanet.errors import EngineNotFoundError

if TYPE_CHECKING:
    from sym_metanet.network import Network


class NodesEngineBase(ABC):
    """Abstract class of a symbolic engine for modelling highway nodes via the METANET
    framework. The methods of this class implement the various equations proposed in the
    framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    @staticmethod
    @abstractmethod
    def get_upstream_flow(q_lasts, beta, betas, q_orig=None, q_dest=None):
        """Computes the upstream flow from a node to a given exiting link, where the
        node can have multiple entering links and multiple exiting links, according to
        [1, Section 3.2.2]. Moreover, this node could be attached to an origin or a
        destination, which is accounted for by the optional parameters `q_orig` and
        `beta_dest`, respectively.

        Parameters
        ----------
        q_lasts
            Flows of the last segments of the links entering the node.
        beta
            Turnrate of the given link exiting from the node.
        betas
            Turnrates of all the exiting links (including this link).
        q_orig : optional
            Optional upstream flow from an origin.
        q_dest : optional
            Optional downstream flow to a off-ramp destination.

        Returns
        -------
        flow
            (virtual) upstream flow in the first segment of the exiting link.
        """

    @staticmethod
    @abstractmethod
    def get_upstream_speed(q_lasts, v_lasts):
        """Computes the upstream speed from a node to a given exiting link, where the
        node can have multiple entering links and multiple exiting links, according to
        [1, Equation 3.10].

        Parameters
        ----------
        q_lasts
            Flows of the last segments of the links entering the node.
        v_lasts
            Speeds of the last segments of the links entering the node.

        Returns
        -------
        speed
            (virtual) upstream speed in the first segment of the exiting link.
        """

    @staticmethod
    @abstractmethod
    def get_downstream_density(rho_firsts):
        """Computes the downstream density from a node to a given entering link, where
        the node can have multiple entering links and multiple exiting links, according
        to [1, Equation 3.9].

        Parameters
        ----------
        rho_firsts
            Densities of the first segments of the links exiting the node.

        Returns
        -------
        speed
            (virtual) downstream density in the last segment of the entering link.
        """


class LinksEngineBase(ABC):
    """Abstract class of a symbolic engine for modelling highway link via the METANET
    framework. The methods of this class implement the various equations proposed in the
    framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    @staticmethod
    @abstractmethod
    def get_flow(rho, v, lanes):
        """Computes the flows of the link's segments, according to [1, Equation 3.1].

        Parameters
        ----------
        rho
            Densities of the link's segments.
        v_free
            Speeds of the link's segments.
        lanes
            Number of lanes in the link.

        Returns
        -------
        flow
            The flow in each link's segment.
        """

    @staticmethod
    @abstractmethod
    def step_density(rho, q, q_up, lanes, L, T):
        """Computes the densities of the link's segments at the next time instant,
        according to [1, Equation 3.2].

        Parameters
        ----------
        rho
            Densities of this link's segments.
        q
            Flows of this link's segments.
        q_up
            Flows of the upstream link's segments.
        lanes
            Number of lanes in the link.
        L
            Link's segment length.
        T
            Sampling time.

        Returns
        -------
        densities_next
            The density in each link's segment at the next time instant.
        """

    @staticmethod
    @abstractmethod
    def step_speed(
        v,
        v_up,
        rho,
        rho_down,
        Veq,
        lanes,
        L,
        tau,
        eta,
        kappa,
        T,
        q_ramp=None,
        delta=None,
        lanes_drop=None,
        phi=None,
        rho_crit=None,
    ):
        """Computes the speeds of the link's segments at the next time instant,
        according to [1, Equation 3.3]. If `q_ramp` and `delta` are provided, then it
        also accounts for merging phenomum [1, Equation 3.7]. Similarly, if `lane_drop`
        and `phi` are provided, then it also accounts for lane drops in the incoming
        link [1, Equation 3.8].

        Parameters
        ----------
        v
            Speeds of this link's segments.
        v_up
            Speeds of the upstream link's segments.
        rho
            Densities of this link's segments.
        rho_down
            Densities of the downstream link's segments.
        Veq
            Equilibrium speed of this link.
        lanes
            Number of lanes in the link.
        L
            Link's segment length.
        tau
            Model parameter for the relaxation term.
        eta
            Model parameter for the anticipation term.
        kappa
            Model parameter for the anticipation term.
        T
            Sampling time.
        q_ramp : optional
            If provided, it represents the flow of an on-ramp attached to the first
            segment of this link, and is used to compute merging phenomenum of on-ramp
            traffic with link traffic.
        delta : optional
            Model parameter for merging phenomenum. Necessary only if `q_ramp` is
            provided as well.
        lanes_drop : optional
            If provided, it represents the difference of lanes of this link minus the
            lanes in the next link, and is used to compute the impact of the lane drop
            on speeds.
        phi : optional
            Model parameter for lane drop phenomenum. Necessary only if `lanes_drop` and
            `rho_crit` are provided as well.
        rho_crit : optional
            Critical density of the link. Necessary only if `lanes_drop` and `phi` are
            provided as well.

        Returns
        -------
        speeds_next
            The speed in each link's segment at the next time instant.
        """

    @staticmethod
    @abstractmethod
    def Veq(rho, v_free, rho_crit, a):
        """Computes the equilibrium speed of the link, according to [1, Equation 3.4].

        Parameters
        ----------
        rho
            Densities of the link's segments.
        v_free
            Free-flow speed of the link.
        rho_crit
            Critical density of the link.
        a
            Model parameter in the equilibrium speed exponent.

        Returns
        -------
        Veq
            The equilibrium speed of the link.
        """

    @staticmethod
    @abstractmethod
    def controlled_Veq(rho, v_ctrl, vsl: list[int], alpha, v_free, rho_crit, a):
        """Computes the equilibrium speed of the link where some of its segments are
        controlled with variable speed limits, according to [1, Equation 3.11].

        Parameters
        ----------
        rho
            Densities of the link's segments.
        v_ctrl
            Speed limits of the controlled segments.
        vsl
            List of indices of the controlled segments.
        alpha
            Non-compliance factor to the indicated speeds.
        v_free
            Free-flow speed of the link.
        rho_crit
            Critical density of the link.
        a
            Model parameter in the equilibrium speed exponent.

        Returns
        -------
        Veq
            The controlled equilibrium speed of the link.
        """


class OriginsEngineBase(ABC):
    """Abstract class of a symbolic engine for modelling highway origins via the METANET
    framework. The methods of this class implement the various equations proposed in the
    framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    @staticmethod
    @abstractmethod
    def step_queue(w, d, q, T):
        """Computes the queue of the origin at the next time instant, according to
        [1, Section 3.2.1].

        Parameters
        ----------
        w
            Queue of this origin.
        d
            Demand at this origin.
        q
            Outflow from this origin.
        T
            Sampling time.

        Returns
        -------
        queue_next
            The queue of the origin at the next time instant.
        """

    @staticmethod
    @abstractmethod
    def get_mainstream_flow(d, w, v_ctrl, v_first, rho_crit, a, v_free, lanes, T):
        """Computes the flow of a mainstream origin according to [1, Section 3.3.3].

        Parameters
        ----------
        d
            Demand at the origin.
        w
            Queue of the origin.
        v_ctrl
            Speed limit of the mainstream origin.
        v_first
            Speed of the first segment of the link the origin is attached to.
        rho_crit
            Critical density of the link the origin is attached to.
        a
            Model parameter (in the equilibrium speed)  of the link the origin is
            attached to.
        v_free
            Free-flow speed of the link the origin is attached to.
        lanes
            Number of lanes in the link the origin is attached to.
        T
            Sampling time.

        Returns
        -------
        flow
            The flow of the mainstream origin.
        """

    @staticmethod
    @abstractmethod
    def get_ramp_flow(
        d, w, C, r, rho_max, rho_first, rho_crit, T, type: Literal["in", "out"] = "out"
    ):
        """Computes the flow of the ramp origin according to [1, Equation 3.5] if
        `type='in'`, or [1, Equation 3.6] if `type='out'`.

        Parameters
        ----------
        d
            Demand at the ramp.
        w
            Queue of the ramp.
        C
            Capacity of the ramp.
        r
            Metering rate (control action) of the ramp.
        rho_max
            Maximum density of the link the ramp is attached to.
        rho_first
            Density of the first segment of the link the ramp is attached to.
        rho_crit
            Critical density of the link the ramp is attached to.
        T
            Sampling time.
        type : 'in' or 'out', optional
            Whether the metering rate `r` should be inside or outside the min function;
            by default, 'out'. See [1, Equations 3.5 and 3.6] for more details.

        Returns
        -------
        flow
            The flow of the ramp.
        """

    @staticmethod
    @abstractmethod
    def get_simplifiedramp_flow(
        qdes,
        d=None,
        w=None,
        C=None,
        rho_max=None,
        rho_first=None,
        rho_crit=None,
        T=None,
        type: Literal["limited", "unlimited"] = "limited",
    ):
        """Computes the flow of a simplified ramp origin.

        Parameters
        ----------
        qdes
            Desired flow at the ramp.
        d : optional
            Demand at the ramp.
        w : optional
            Queue of the ramp.
        C : optional
            Capacity of the ramp.
        rho_first : optional
            Density of the first segment of the link the ramp is attached to. To be
            provided only if `type=limited`.
        rho_max : optional
            Maximum density of the link the ramp is attached to. To be provided only if
            `type=limited`.
        rho_crit : optional
            Critical density of the link the ramp is attached to. To be provided only if
            `type=limited`.
        T : optional
            Sampling time.
        type : "limited" or "unlimited", optional
            Type of equation to compute the flow for a simplified metered ramp. By
            default, "limited" is selected, so that the desired flow is limited
            according to the capacity, current demands, queues and densities of the ramp
            and its attached segment. This choice requires that all optional arguments
            are provided. Otherwise, "unlimited" makes sure that the  desired flow is
            the actual flow of the ramp, which nevertheless can cause negative queues.

        Returns
        -------
        flow
            The flow of the ramp.
        """


class DestinationsEngineBase(ABC):
    """Abstract class of a symbolic engine for modelling highway destinations via the
    METANET framework. The methods of this class implement the various equations
    proposed in the framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    @staticmethod
    @abstractmethod
    def get_congestion_free_downstream_density(rho_last, rho_crit):
        """For a link with a downstream destination with no congestion, i.e., ideal,
        returns the downstream density.

        Parameters
        ----------
        rho_last
            Density in the link's last segment.
        rho_crit
            Critical density of the link entering the destination.

        Returns
        -------
        density
            The downstream density for a link connected to a congestion-free
            destination.
        """

    @staticmethod
    @abstractmethod
    def get_congested_downstream_density(rho_last, rho_destination, rho_crit):
        """For a link with a downstream congested destination, returns the downstream
        density.

        Parameters
        ----------
        rho_last
            Density in the link's last segment.
        rho_destination
            Density scenario of the congested destination.
        rho_crit
            Critical density of the link entering the destination.

        Returns
        -------
        density
            The downstream density for a link connected to a destination.
        """


class EngineBase(ABC):
    """Abstract class of a symbolic engine for modelling highways via the METANET
    framework. The methods of this class implement the various equations proposed in the
    framework, which can be found in [1].

    References
    ----------
    [1] Hegyi, A., 2004, "Model predictive control for integrating traffic control
        measures", Netherlands TRAIL Research School.
    """

    @property
    @abstractmethod
    def nodes(self) -> type[NodesEngineBase]:
        pass

    @property
    @abstractmethod
    def links(self) -> type[LinksEngineBase]:
        pass

    @property
    @abstractmethod
    def origins(self) -> type[OriginsEngineBase]:
        pass

    @property
    @abstractmethod
    def destinations(self) -> type[DestinationsEngineBase]:
        pass

    @abstractmethod
    def var(self, name: str, n: int = 1, *args, **kwargs):
        """Creates a variable.

        Parameters
        ----------
        name : str
            Name of the variable.
        n : int
            Length of the variable.

        Returns
        -------
        sym variable
            The symbolic variable.
        """

    @abstractmethod
    def vcat(self, *arrays):
        """Concatenates vertically two elements/variables.

        Parameters
        ----------
        arrays
            Values to be vertically concatenated.

        Returns
        -------
        sym variable
            A unique variable resulting from the vertical concatenation.
        """

    @abstractmethod
    def min(self, array1, array2):
        """Returns the minimum value of the given elements/variables.

        Parameters
        ----------
        array1, array2
            Values to be compared.

        Returns
        -------
        sym variable
            The minimum value.
        """

    @abstractmethod
    def max(self, array1, array2):
        """Returns the maximum value of the given elements/variables.

        Parameters
        ----------
        array1, array2
            Values to be compared.

        Returns
        -------
        sym variable
            The maximum value.
        """

    @abstractmethod
    def to_function(self, net: "Network", *args, **kwargs) -> Callable:
        """Returns the network's dynamics as a callable function, with different nature
        depending on the engine implementation.

        Parameters
        ----------
        net : Network
            The network whose dynamics must be translated into a function.

        Returns
        -------
        Callable
            The function representing the dynamics.
        """


def get_current_engine() -> EngineBase:
    """Gets the current symbolic engine.

    Returns
    -------
    EngineBase
        The current symbolic engine.
    """
    return sym_metanet.engine


def get_available_engines() -> dict[str, dict[str, str]]:
    """Returns the available symbolic engines for METANET modelling, for which an
    implementation exists.

    Returns
    -------
    Dict[str, Dict[str, str]]
        The available engines in the form of a `dict` whose keys are the available
        engine class types, and the values are info on each engine, such as module and
        class name.
    """
    return {
        "casadi": {"module": "sym_metanet.engines.casadi", "class": "Engine"},
        "numpy": {"module": "sym_metanet.engines.numpy", "class": "Engine"},
    }


def use(engine: Union[str, EngineBase], *args, **kwargs) -> EngineBase:
    """Uses the given symbolic engine for computations.

    Parameters
    ----------
    engine : str or instance of engine
        If a string, then `engine` must represent the class name of one of the available
        engines, so that it can be instantiated with args and kwargs (see
        `sym_metanet.engines.get_available_engines`). Otherwise, it must be an instance
        of an engine itself that inherits from `sym_metanet.engines.EngineBase`.
    args, kwargs
        Passed to the engine constructor in case `engine` is a string.

    Returns
    -------
    EngineBase
        A reference to the new engine, if `engine` is a string, or the reference to the
        same instance, otherwise.

    Raises
    ------
    ValueError
        Raises in case `engine` is neither a string nor a `EngineBase` instance.
    EngineNotFoundError
        Raises in case `engine` is a string but matches no available engines.
    """
    if isinstance(engine, EngineBase):
        sym_metanet.engine = engine
    else:
        engines = get_available_engines()
        if engine not in engines:
            raise EngineNotFoundError(
                f'Engine class must be in {{{", ".join(engines)}}}; got '
                f"{engine} instead."
            )
        from importlib import import_module

        engineinfo = engines[engine]
        cls = getattr(import_module(engineinfo["module"]), engineinfo["class"])
        sym_metanet.engine = cls(*args, **kwargs)
    return sym_metanet.engine
