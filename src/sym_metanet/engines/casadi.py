from itertools import chain, product
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, TypeVar, Union

import casadi as cs

from sym_metanet.blocks.base import ElementWithVars
from sym_metanet.engines.core import (
    DestinationsEngineBase,
    EngineBase,
    LinksEngineBase,
    NodesEngineBase,
    OriginsEngineBase,
)

if TYPE_CHECKING:
    from sym_metanet.blocks.links import Link
    from sym_metanet.network import Network


VarType = TypeVar("VarType", cs.SX, cs.MX)


class NodesEngine(NodesEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.NodesEngineBase`."""

    @staticmethod
    def get_upstream_flow(
        q_lasts: VarType,
        beta: VarType,
        betas: VarType,
        q_orig: Optional[VarType] = None,
        q_dest: Optional[VarType] = None,
    ) -> VarType:
        Q = cs.sum1(q_lasts)
        if q_orig is not None:
            Q += q_orig
        if q_dest is not None:
            Q -= q_dest
        return (beta / cs.sum1(betas)) * Q

    @staticmethod
    def get_upstream_speed(q_lasts: VarType, v_lasts: VarType) -> VarType:
        return cs.sum1(v_lasts * q_lasts) / cs.sum1(q_lasts)

    @staticmethod
    def get_downstream_density(rho_firsts: VarType) -> VarType:
        return cs.sum1(rho_firsts**2) / cs.sum1(rho_firsts)


class LinksEngine(LinksEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.LinksEngineBase`."""

    @staticmethod
    def get_flow(rho: VarType, v: VarType, lanes: VarType) -> VarType:
        return rho * v * lanes

    @staticmethod
    def step_density(
        rho: VarType, q: VarType, q_up: VarType, lanes: VarType, L: VarType, T: VarType
    ) -> VarType:
        return rho + (T / lanes / L) * (q_up - q)

    @staticmethod
    def step_speed(
        v: VarType,
        v_up: VarType,
        rho: VarType,
        rho_down: VarType,
        Veq: VarType,
        lanes: VarType,
        L: VarType,
        tau: VarType,
        eta: VarType,
        kappa: VarType,
        T: VarType,
        q_ramp: Optional[VarType] = None,
        delta: Optional[VarType] = None,
        lanes_drop: Optional[VarType] = None,
        phi: Optional[VarType] = None,
        rho_crit: Optional[VarType] = None,
    ) -> VarType:
        relaxation = (T / tau) * (Veq - v)
        convection = T * v / L * (v_up - v)
        anticipation = (eta * T / tau) * (rho_down - rho) / (L * (rho + kappa))
        v_next = v + relaxation + convection - anticipation
        if q_ramp is not None and delta is not None:
            v_next[0] -= (delta * T * q_ramp * v[0]) / (L * lanes * (rho[0] + kappa))
        if lanes_drop is not None and phi is not None and rho_crit is not None:
            v_next[-1] -= (phi * T * lanes_drop * rho[-1] * v[-1] ** 2) / (
                L * lanes * rho_crit
            )
        return v_next

    @staticmethod
    def Veq(rho: VarType, v_free: VarType, rho_crit: VarType, a: VarType) -> VarType:
        return v_free * cs.exp((-1 / a) * cs.power(rho / rho_crit, a))

    @staticmethod
    def controlled_Veq(
        rho: VarType,
        v_ctrl: VarType,
        vsl: list[int],
        alpha: VarType,
        v_free: VarType,
        rho_crit: VarType,
        a: VarType,
    ) -> VarType:
        Veq = LinksEngine.Veq(rho, v_free, rho_crit, a)
        Veq[vsl] = cs.fmin(Veq[vsl], (1 + alpha) * v_ctrl)
        return Veq


class OriginsEngine(OriginsEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.OriginsEngineBase`."""

    @staticmethod
    def step_queue(w: VarType, d: VarType, q: VarType, T: VarType) -> VarType:
        return w + T * (d - q)

    @staticmethod
    def get_mainstream_flow(
        d: VarType,
        w: VarType,
        v_ctrl: VarType,
        v_first: VarType,
        rho_crit: VarType,
        a: VarType,
        v_free: VarType,
        lanes: VarType,
        T: VarType,
    ) -> VarType:
        V_crit = LinksEngine.Veq(rho_crit, v_free, rho_crit, a)
        v_lim = cs.fmin(v_ctrl, v_first)
        ratio = v_lim / v_free
        ratio = cs.fmax(0.05, cs.fmin(1.0, ratio))  # limit ratio to avoid nans
        q_speed = lanes * v_lim * rho_crit * cs.power(-a * cs.log(ratio), 1 / a)
        q_cap = lanes * V_crit * rho_crit
        q_lim = cs.if_else(v_lim < V_crit, q_speed, q_cap)
        return cs.fmin(d + w / T, q_lim)

    @staticmethod
    def get_ramp_flow(
        d: VarType,
        w: VarType,
        C: VarType,
        r: VarType,
        rho_max: VarType,
        rho_first: VarType,
        rho_crit: VarType,
        T: VarType,
        type: Literal["in", "out"] = "out",
    ) -> VarType:
        term1 = d + w / T
        term3 = (rho_max - rho_first) / (rho_max - rho_crit)
        if type == "in":
            return cs.fmin(term1, C * cs.fmin(r, term3))
        return r * cs.fmin(term1, C * cs.fmin(1, term3))

    @staticmethod
    def get_simplifiedramp_flow(
        qdes: VarType,
        d: VarType = None,
        w: VarType = None,
        C: VarType = None,
        rho_max: VarType = None,
        rho_first: VarType = None,
        rho_crit: VarType = None,
        T: VarType = None,
        type: Literal["limited", "unlimited"] = "limited",
    ) -> VarType:
        if type == "unlimited":
            return qdes
        term2 = d + w / T
        term3 = C * cs.fmin(1, (rho_max - rho_first) / (rho_max - rho_crit))
        return cs.fmin(qdes, cs.fmin(term2, term3))


class DestinationsEngine(DestinationsEngineBase, Generic[VarType]):
    """CasADi implementation of `sym_metanet.engines.core.DestinationsEngineBase`."""

    @staticmethod
    def get_congestion_free_downstream_density(
        rho_last: VarType, rho_crit: VarType
    ) -> VarType:
        return cs.fmin(rho_last, rho_crit)

    @staticmethod
    def get_congested_downstream_density(
        rho_last: VarType, rho_destination: VarType, rho_crit: VarType
    ) -> VarType:
        return cs.fmax(cs.fmin(rho_last, rho_crit), rho_destination)


class Engine(EngineBase, Generic[VarType]):
    """Symbolic engine implemented with the CasADi framework"""

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        """Instantiates a CasADi engine.

        Parameters
        ----------
        sym_type : {'SX', 'MX'}, optional
            A string that tells the engine with type of symbolic variables to use. Must
            be either `'SX'` or `'MX'`, at which point the engine employes `casadi.SX`
            or `casadi.MX` variables, respectively. By default, `'SX'` is used.

        Raises
        ------
        AttributeError
            Raises if `sym_type` is not valid.
        """
        super().__init__()
        self.sym_type: Union[type[cs.SX], type[cs.MX]] = getattr(cs, sym_type)

    @property
    def nodes(self) -> type[NodesEngine[VarType]]:
        return NodesEngine[VarType]

    @property
    def links(self) -> type[LinksEngine[VarType]]:
        return LinksEngine[VarType]

    @property
    def origins(self) -> type[OriginsEngine[VarType]]:
        return OriginsEngine[VarType]

    @property
    def destinations(self) -> type[DestinationsEngine[VarType]]:
        return DestinationsEngine[VarType]

    def var(self, name: str, n: int = 1, *args, **kwargs) -> VarType:
        return self.sym_type.sym(name, n, 1)

    def vcat(self, *arrays: VarType) -> VarType:
        return cs.vcat(arrays)

    def min(self, array1: VarType, array2: VarType) -> VarType:
        return cs.fmin(array1, array2)

    def max(self, array1: VarType, array2: VarType) -> VarType:
        return cs.fmax(array1, array2)

    def to_function(  # type: ignore[override]
        self,
        net: "Network",
        compact: int = 0,
        more_out: bool = False,
        parameters: Optional[dict[str, VarType]] = None,
        **other_parameters: Any,
    ) -> cs.Function:
        """Converts the network's dynamics to a CasADi Function.

        Parameters
        ----------
        net : Network
            The network whose dynamics must be translated into a function.
        compact : int, optional
            The compactness of input and output arguments. The levels are

            - <= 0: no aggregation of arguments, i.e., the function keeps states, action
            or disturbances for each element separate.

            - == 1: some aggregation, i.e., same variable types are clumped together.

            -  > 1: most aggregation, i.e., states, action and disturbances are
            aggregated in a single vector each.

        more_out : bool, optional
            Includes flows of links, origins and destinations (if any) in the output. By
            default `False`.
        parameters : dict[str, casadi.SX or MX], optional
            Symbolic network parameters to be included in the function, by default None.
        **other_parameters
            Other parameters (numerical or symbolical) required during the computations,
            e.g., sampling time T is usually required.

        Returns
        -------
        cs.Function
            The CasADi Function representing the network's dynamics.

        Raises
        ------
        RuntimeError
            Raises if variables have not yet been initialized; or if the dynamics have
            not been stepped yet, so no state at the next time instant is found.
        """
        for el, group in product(
            net.elements, ["_states", "_actions", "_disturbances"]
        ):
            if any(getattr(el, group)) and not getattr(el, f"has{group}"):
                raise RuntimeError(
                    f"Found no {group[1:-1]} in {el.name}; perhaps variables "
                    "have not been initialized via `net.init_vars`?"
                )
            if any(el._states) and not el.has_next_states:
                raise RuntimeError(
                    f"Found no next state in {el.name}; perhaps dynamics have "
                    "not been stepped via `net.step`?"
                )

        if parameters is None:
            parameters = {}

        # gather inputs
        x = {el: _filter_vars(vars) for el, vars in net.states.items()}
        u = {el: _filter_vars(vars) for el, vars in net.actions.items()}
        d = {el: _filter_vars(vars) for el, vars in net.disturbances.items()}
        names_in, args_in = _gather_inputs(x, u, d, compact)
        if parameters:
            _add_parameters_to_inputs(names_in, args_in, parameters, compact)

        # gather outputs
        x_next = {
            el: _filter_vars(vars, independent=False)
            for el, vars in net.next_states.items()
        }
        names_out, args_out = _gather_outputs(x_next, compact)
        if more_out:
            _add_flows_to_outputs(
                names_out, args_out, self, net, parameters, other_parameters, compact
            )

        # create dynamics function
        return cs.Function(
            "F",
            args_in,
            args_out,
            names_in,
            names_out,
            {"allow_duplicate_io_names": True, "cse": True},
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(casadi)"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(casadi, type={self.sym_type.__name__})"


def _filter_vars(
    vars: dict[str, Union[VarType, Any]], independent: bool = True
) -> dict[str, VarType]:
    """Internal utility to filter out symbols that are either only symbolic
    and/or independent (and thus can be inputs to `casadi.Function`)."""

    def filter(var: Union[VarType, Any]) -> Optional[VarType]:
        if isinstance(var, cs.SX):
            if not independent or all(var[i].n_dep() == 0 for i in range(var.size1())):
                return var
            varlists = cs.symvar(var)
            if len(varlists) == var.size1():
                return cs.vcat(varlists)

        if isinstance(var, cs.MX):
            if not independent or var.n_dep() == 0:
                return var
            varlists = cs.symvar(var)
            if len(varlists) == 1:
                return varlists[0]

        return None

    filtered_vars = {}
    for name, var in vars.items():
        filtered = filter(var)
        if filtered is not None:
            filtered_vars[name] = filtered
    return filtered_vars


def _gather_inputs(
    x: dict[ElementWithVars, dict[str, VarType]],
    u: dict[ElementWithVars, dict[str, VarType]],
    d: dict[ElementWithVars, dict[str, VarType]],
    compact: int,
) -> tuple[list[str], list[VarType]]:
    """Internal utility to gather inputs for `casadi.Function`."""

    if compact <= 0:
        # no aggregation
        names_in, args_in = [], []
        for vars_in in (x, u, d):
            for el, vars in vars_in.items():  # type: ignore[attr-defined]
                for varname, var in vars.items():
                    names_in.append(f"{varname}_{el.name}")
                    args_in.append(var)
        return names_in, args_in

    # group variables as (name, list of vars)
    states: dict[str, list[VarType]] = {}
    actions: dict[str, list[VarType]] = {}
    disturbances: dict[str, list[VarType]] = {}
    for vars_in, group in [(x, states), (u, actions), (d, disturbances)]:
        for el, vars in vars_in.items():  # type: ignore[attr-defined]
            for varname, var in vars.items():
                if varname in group:
                    group[varname].append(var)
                else:
                    group[varname] = [var]

    # group variables as (name, symbol)
    for group in (states, actions, disturbances):
        for varname, list_of_vars in group.items():
            group[varname] = cs.vcat(list_of_vars)

    # add to names and args
    if compact == 1:
        names_in = list(chain(states.keys(), actions.keys(), disturbances.keys()))
        args_in = list(chain(states.values(), actions.values(), disturbances.values()))
    else:
        names_in = ["x", "u", "d"]
        args_in = [
            cs.vcat(states.values()),
            cs.vcat(actions.values()),
            cs.vcat(disturbances.values()),
        ]
    return names_in, args_in


def _gather_outputs(
    x_next: dict[ElementWithVars, dict[str, VarType]],
    compact: int,
) -> tuple[list[str], list[VarType]]:
    """Internal utility to gather outputs for `casadi.Function`."""

    if compact <= 0:
        # no aggregation
        names_out, args_out = [], []
        for el, vars in x_next.items():
            for varname, var in vars.items():
                names_out.append(f"{varname}_{el.name}+")
                args_out.append(var)
        return names_out, args_out

    # group variables as (name, list of vars)
    next_states: dict[str, list[VarType]] = {}
    for vars in x_next.values():
        for varname, var in vars.items():
            varname += "+"
            if varname in next_states:
                next_states[varname].append(var)
            else:
                next_states[varname] = [var]

    # group variables as (name, symbol)
    for varname, list_of_vars in next_states.items():
        next_states[varname] = cs.vcat(list_of_vars)

    # add to names and args
    if compact == 1:
        names_out = list(next_states.keys())
        args_out = list(next_states.values())
    else:
        names_out = ["x+"]
        args_out = [cs.vcat(next_states.values())]
    return names_out, args_out


def _add_parameters_to_inputs(
    names_in: list[str],
    args_in: list[VarType],
    parameters: dict[str, VarType],
    compact: int,
) -> None:
    """Internal utility to add parameters to inputs for `casadi.Function`."""
    if compact <= 0:
        names_in.extend(parameters.keys())
        args_in.extend(parameters.values())
    else:
        names_in.append("p")
        args_in.append(cs.vcat(parameters.values()))


def _add_flows_to_outputs(
    names_out: list[str],
    args_out: list[VarType],
    engine: Engine,
    net: "Network",
    parameters: dict[str, VarType],
    other_parameters: dict[str, Any],
    compact: int,
) -> None:
    """Internal utility to add even more outputs for `casadi.Function`."""

    # add link flows (q), origin flows (q_o) and offramp flows (q_d) to output
    names_link: list[str] = []
    flows_link: list[VarType] = []
    names_origins, flows_origins = [], []
    names_offramps, flows_offramps = [], []
    for _, _, link in net.links:
        names_link.append(f"q_{link.name}")
        flows_link.append(link.get_flow(engine))
    for origin in net.origins:
        names_origins.append(f"q_o_{origin.name}")
        flows_origins.append(
            origin.get_flow(net, engine=engine, **parameters, **other_parameters)
        )
    for destination in net.destinations:
        if hasattr(destination, "get_flow"):
            names_offramps.append(f"q_d_{destination.name}")
            flows_offramps.append(
                destination.get_flow(
                    net, engine=engine, **parameters, **other_parameters
                )
            )

    if compact > 0:
        names_link = ["q"]
        flows_link = [cs.vcat(flows_link)]
        if names_origins:
            names_origins = ["q_o"]
            flows_origins = [cs.vcat(flows_origins)]
        else:
            names_origins, flows_origins = [], []
        if names_offramps:
            names_offramps = ["q_d"]
            flows_offramps = [cs.vcat(flows_offramps)]
        else:
            names_offramps, flows_offramps = [], []
    if compact > 1:
        names_link = ["q"]
        flows_link = [cs.vertcat(flows_link[0], flows_origins[0])]
        names_origins, flows_origins = [], []
        names_offramps, flows_offramps = [], []

    names_out.extend(names_link + names_origins + names_offramps)
    args_out.extend(flows_link + flows_origins + flows_offramps)
