import unittest
from typing import Literal, Union

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

import sym_metanet as metanet
from sym_metanet import (
    CongestedDestination,
    Link,
    MeteredOnRamp,
    Network,
    Node,
    OffRamp,
    SimplifiedMeteredOnRamp,
    engines,
)
from sym_metanet.engines.casadi import Engine as CasadiEngine
from sym_metanet.engines.numpy import Engine as NumpyEngine
from sym_metanet.errors import EngineNotFoundError
from sym_metanet.util.funcs import first


def get_net(
    sym_type: Literal["MX", "SX"], link_with_ramp: int = 2, offramp: bool = False
) -> tuple[Network, dict[str, Union[cs.SX, cs.MX]], dict[str, float]]:
    assert link_with_ramp in {1, 2}
    XX: Union[type[cs.SX], type[cs.MX]] = getattr(cs, sym_type)
    L = 1
    lanes = 2
    C = (3500, 2000)
    tau = 18 / 3600
    kappa = 40
    eta = 60
    rho_max = 180
    delta = 0.0122
    T = 10 / 3600
    a_sym = XX.sym("a")
    v_free_sym = XX.sym("v_free")
    rho_crit_sym = XX.sym("rho_crit_sym")
    N1 = Node(name="N1")
    N2 = Node(name="N2")
    N3 = Node(name="N3")
    O1 = MeteredOnRamp(C[0], name="O1")
    D1 = CongestedDestination(name="D1")
    if link_with_ramp == 1:
        L1 = Link(1, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L1")
        L2 = Link(2, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L2")
    else:
        L1 = Link(2, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L1")
        L2 = Link(1, lanes, L, rho_max, rho_crit_sym, v_free_sym, a_sym, name="L2")
    net = Network(name="A1").add_path(
        origin=O1, path=(N1, L1, N2, L2, N3), destination=D1
    )
    if not offramp:
        O2 = SimplifiedMeteredOnRamp(C[1], name="O2")
        net.add_origin(O2, N2)
    else:
        D2 = OffRamp(turnrate=0.5, name="D2")
        net.add_destination(D2, N2)
    sym_pars = {"rho_crit": rho_crit_sym, "a": a_sym, "v_free": v_free_sym}
    others = {
        "L": L,
        "lanes": lanes,
        "C": C,
        "rho_max": rho_max,
        "T": T,
        "tau": tau,
        "eta": eta,
        "kappa": kappa,
        "delta": delta,
    }
    return net, sym_pars, others


def get_hardcoded_dynamics(
    L, lanes, C, tau, kappa, eta, rho_max, delta, T, link_with_ramp: int = 2, **_
) -> cs.Function:
    # sourcery skip: use-contextlib-suppress
    assert link_with_ramp in {1, 2}
    rho = cs.SX.sym("rho", 3, 1)
    v = cs.SX.sym("v", 3, 1)
    w = cs.SX.sym("w", 2, 1)
    r = cs.SX.sym("r")
    q_O2 = cs.SX.sym("q_O2")
    d = cs.SX.sym("d", 3, 1)
    rho_crit = cs.SX.sym("rho_crit")
    a = cs.SX.sym("a")
    v_free = cs.SX.sym("v_free")

    q_O1 = r * cs.fmin(
        d[0] + w[0] / T, C[0] * cs.fmin(1, (rho_max - rho[0]) / (rho_max - rho_crit))
    )
    q_O2_limited = cs.mmin(
        cs.vertcat(  # must limit the desired q_O2 to be feasible
            q_O2,
            w[1] / T + d[1],
            C[1],
            C[1] * (rho_max - rho[1]) / (rho_max - rho_crit),
        ),
    )
    q_o = cs.vertcat(q_O1, q_O2_limited)
    w_next = w + T * (d[:2] - q_o)

    q = lanes * rho * v
    q_up = cs.vertcat(q_O1, q[0], q[1])
    q_up[link_with_ramp] += q_O2_limited
    v_up = cs.vertcat(v[0], v[0], v[1])
    rho_down = cs.vertcat(rho[1], rho[2], cs.fmax(cs.fmin(rho[2], rho_crit), d[2]))
    Veq = v_free * cs.exp((-1 / a) * (rho / rho_crit) ** a)
    rho_next = rho + (T / (L * lanes)) * (q_up - q)
    v_next = (
        v
        + T / tau * (Veq - v)
        + T / L * v * (v_up - v)
        - eta * T / tau / L * (rho_down - rho) / (rho + kappa)
    )
    merging_num = delta * T * q_O2_limited * v[link_with_ramp]
    merging_den = L * lanes * (rho[link_with_ramp] + kappa)
    v_next[link_with_ramp] -= merging_num / merging_den

    pars = cs.vertcat(rho_crit, a, v_free)
    # rho_next = cs.fmax(0, rho_next)
    v_next = cs.fmax(0, v_next)
    # w_next = cs.fmax(0, w_next)
    # q = cs.fmax(0, q)
    # q_o = cs.fmax(0, q_o)
    return cs.Function(
        "F", [rho, v, w, r, q_O2, d, pars], [rho_next, v_next, w_next, q, q_o]
    )


class TestEngines(unittest.TestCase):
    def test_get_current_engine__gets_correct_engine(self):
        self.assertIs(engines.get_current_engine(), metanet.engine)

    def test_get_available_engines__gets_available_engines_correctly(self):
        engines_ = engines.get_available_engines()
        self.assertIsInstance(engines_, dict)
        self.assertIsInstance(first(engines_.keys()), str)

    def test_use__sets_the_engine_correctly(self):
        old_engine = metanet.engine
        new_engine = engines.use("casadi")
        self.assertIsNot(metanet.engine, old_engine)
        self.assertIs(metanet.engine, new_engine)

    def test_use__raises__when_engine_not_found(self):
        invalid_engine = object()
        with self.assertRaises(EngineNotFoundError):
            engines.use(invalid_engine)


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestCasadiEngine(unittest.TestCase):
    def test_to_function__fails__with_uninit_vars(self):
        net, sym_pars, other_pars = get_net(self.sym_type)
        self.assertTrue(net.is_valid(raises=False)[0])
        engine = engines.use("casadi", sym_type=self.sym_type)

        with self.assertRaises(RuntimeError):
            engine.to_function(
                net=net,
                parameters=sym_pars,
                T=other_pars["T"],
                compact=1,
                more_out=True,
            )

    def test_to_function__fails__with_unstepped_dynamics(self):
        net, sym_pars, other_pars = get_net(self.sym_type)
        self.assertTrue(net.is_valid(raises=False)[0])
        engine = engines.use("casadi", sym_type=self.sym_type)
        for el in net.elements:
            el.init_vars(engine=engine)
        with self.assertRaises(RuntimeError):
            engine.to_function(
                net=net,
                parameters=sym_pars,
                T=other_pars["T"],
                compact=1,
                more_out=True,
            )

    def test_to_function__works__with_positive_init_and_next_quantities(self):
        net, sym_pars, other_pars = get_net(self.sym_type)
        engine = engines.use("casadi", sym_type=self.sym_type)
        net.is_valid(raises=True)
        net.step(positive_init_density=True, engine=engine, **other_pars)
        F = metanet.engine.to_function(
            net=net,
            **other_pars,
            parameters=sym_pars,
            more_out=True,
            compact=1,
        )
        self.assertIn("rho", F.name_in())
        self.assertFalse(F.get_free())
        self.assertEqual(F.n_in(), 7)
        self.assertEqual(F.n_out(), 5)

    @parameterized.expand([(1,), (2,)])
    def test_to_function__numerically_works(self, link_with_ramp: int):
        net, sym_pars, other_pars = get_net(self.sym_type, link_with_ramp)
        engine = engines.use("casadi", sym_type=self.sym_type)
        net.is_valid(raises=True)
        net.step(positive_next_speed=True, engine=engine, **other_pars)
        Fact: cs.Function = metanet.engine.to_function(
            net=net,
            **other_pars,
            parameters=sym_pars,
            more_out=True,
            compact=1,
        )
        Fexp = get_hardcoded_dynamics(**other_pars, link_with_ramp=link_with_ramp)

        np_random = np.random.default_rng(69420)
        names = Fact.name_out()
        for i in range(1_000):
            if i == 0:
                rho = np.asarray([10, 10, 10])
                v = np.asarray([100, 100, 100])
                w = np.asarray([0, 0])
                r = 1.0
                q = 575.0
                d = np.asarray([1027.85313984, 391.81732715, 21.76034038])
                a = 1.867
                v_free = 102.0
                rho_crit = 33.5
            else:
                rho = np_random.normal(30, 10, 3)
                v = np_random.normal(100, 30, 3)
                w = np_random.normal(100, 50, 2)
                r = np_random.random()
                q = np_random.normal(800, 200)
                d = np_random.normal((1e3, 1e3, 50), (200, 200, 20))
                rho_crit = np_random.uniform(10, 30)
                a = np_random.uniform(1, 2)
                v_free = np_random.uniform(40, 80)
            args = [
                np.maximum(0, arg)
                for arg in [rho, v, w, r, q, d, [rho_crit, a, v_free]]
            ]
            out_exp = Fexp(*args)
            out_act = Fact(*args)

            for name, exp_, act_ in zip(names, out_exp, out_act):
                exp: np.ndarray = exp_.full().flatten()
                act: np.ndarray = act_.full().flatten()
                self.assertTrue((exp >= 0).all(), f"[exp] {name} negative (iter {i})")
                self.assertTrue((act >= 0).all(), f"[act] {name} negative (iter {i})")
                np.testing.assert_allclose(
                    act,
                    exp,
                    atol=1e-6,
                    rtol=1e-6,
                    err_msg=f"{name} dissimilar (iter {i})",
                )

    def test_to_function__with_offramp__includes_offramp_flow(self):
        net, sym_pars, other_pars = get_net(self.sym_type, offramp=True)
        engine = engines.use("casadi", sym_type=self.sym_type)
        net.is_valid(raises=True)
        net.step(engine=engine, **other_pars)
        F = metanet.engine.to_function(
            net=net,
            **other_pars,
            parameters=sym_pars,
            more_out=True,
            compact=1,
        )
        self.assertIn("q_d", F.name_out())
        self.assertFalse(F.get_free())
        self.assertEqual(F.n_in(), 6)
        self.assertEqual(F.n_out(), 6)


class TestCasadiVsNumpyEngine(unittest.TestCase):
    N = 7
    NE = NumpyEngine()
    CE = CasadiEngine[cs.DM]()

    def test_nodes__get_upstream_flow(self):
        N = self.N
        q_lasts = np.maximum(0, np.random.randn(N) * 50 + 200)
        betas = np.random.randint(low=1, high=10, size=N)
        q_orig = np.maximum(0, np.random.randn() * 50 + 200)
        args = [q_lasts, betas[0], betas, q_orig]
        np.testing.assert_allclose(
            self.NE.nodes.get_upstream_flow(*args),
            self.CE.nodes.get_upstream_flow(*map(cs.DM, args)).full().squeeze(),
        )

    def test_nodes__get_upstream_speed(self):
        N = self.N
        q_lasts = np.maximum(0, np.random.randn(N) * 50 + 200)
        v_lasts = np.maximum(0, np.random.randn(N) * 10 + 60)
        args = [q_lasts, v_lasts]
        np.testing.assert_allclose(
            self.NE.nodes.get_upstream_speed(*args),
            self.CE.nodes.get_upstream_speed(*map(cs.DM, args)).full().squeeze(),
        )

    def test_nodes__get_downstream_density(self):
        N = self.N
        rho_firsts = np.maximum(0, np.random.randn(N) * 5 + 30)
        np.testing.assert_allclose(
            self.NE.nodes.get_downstream_density(rho_firsts),
            self.CE.nodes.get_downstream_density(cs.DM(rho_firsts)).full().squeeze(),
        )

    def test_links__get_flow(self):
        N = self.N
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        v = np.maximum(0, np.random.randn(N) * 10 + 80)
        lanes = np.random.randint(3, 7)
        args = [rho, v, lanes]
        np.testing.assert_allclose(
            self.NE.links.get_flow(*args),
            self.CE.links.get_flow(*map(cs.DM, args)).full().squeeze(),
        )

    def test_links__step_density(self):
        N = self.N
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        q = np.maximum(0, np.random.randn(N) * 10 + 800)
        q_up = np.maximum(0, np.random.randn(N) * 10 + 800)
        lanes = np.random.randint(3, 7)
        L = np.random.rand() * 1000 + 1000
        T = np.random.rand()
        args = [rho, q, q_up, lanes, L, T]
        np.testing.assert_allclose(
            self.NE.links.step_density(*args),
            self.CE.links.step_density(*map(cs.DM, args)).full().squeeze(),
        )

    def test_links__step_speed(self):
        N = self.N
        v = np.maximum(0, np.random.randn(N) * 10 + 100)
        v_up = np.maximum(0, np.random.randn(N) * 10 + 100)
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        rho_down = np.maximum(0, np.random.randn(N) * 5 + 30)
        Veq = np.maximum(0, np.random.randn(N) * 10 + 100)
        lanes = np.random.randint(3, 7)
        L = np.random.rand() * 1000 + 1000
        tau = np.random.rand()
        eta = T = np.random.rand()
        kappa = np.random.rand()
        T = np.random.rand()
        q_ramp = np.maximum(0, np.random.randn() * 100 + 800)
        delta = np.random.rand()
        lanes_drop = np.random.randint(1, 3)
        phi = np.random.rand()
        rho_crit = np.random.rand() * 10 + 30
        args = [
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
            q_ramp,
            delta,
            lanes_drop,
            phi,
            rho_crit,
        ]
        np.testing.assert_allclose(
            self.NE.links.step_speed(*args),
            self.CE.links.step_speed(*map(cs.DM, args)).full().squeeze(),
        )

    def test_links__Veq(self):
        N = self.N
        rho = np.maximum(0, np.random.randn(N) * 5 + 30)
        v_free = np.random.rand() * 40 + 80
        rho_crit = np.random.rand() * 10 + 30
        a = np.random.rand() + 1
        args = [rho, v_free, rho_crit, a]
        np.testing.assert_allclose(
            self.NE.links.Veq(*args),
            self.CE.links.Veq(*map(cs.DM, args)).full().squeeze(),
        )

    def test_origins__step_queue(self):
        w = np.random.randint(low=1, high=3) * 100 + 300
        d = np.random.randint(low=1, high=3) * 100 + 300
        q = np.maximum(0, np.random.randn() * 200 + 800)
        T = np.random.rand()
        args = [w, d, q, T]
        np.testing.assert_allclose(
            self.NE.origins.step_queue(*args),
            self.CE.origins.step_queue(*map(cs.DM, args)).full().squeeze(),
        )

    def test_origins__get_ramp_flow(self):
        w = np.random.randint(low=1, high=3) * 100 + 300
        d = np.random.randint(low=1, high=3) * 100 + 300
        C = np.maximum(100, np.random.randn() * 500 + 2000)
        r = np.random.rand()
        rho_max = (np.random.rand() * 30 + 130) * 2
        rho_first = np.random.rand() * 20 + 20
        rho_crit = np.random.rand() * 20 + 20
        T = np.random.rand()
        args = [d, w, C, r, rho_max, rho_first, rho_crit, T]
        for type_ in ("in", "out"):
            np.testing.assert_allclose(
                self.NE.origins.get_ramp_flow(*args, type_),
                self.CE.origins.get_ramp_flow(*map(cs.DM, args), type_)
                .full()
                .squeeze(),
            )

    def test_destinations__get_ramp_flow(self):
        rho_last = np.random.rand() * 20 + 20
        rho_destination = np.random.rand() * 30 + 20
        rho_crit = np.random.rand() * 20 + 20
        args = [rho_last, rho_destination, rho_crit]
        np.testing.assert_allclose(
            self.NE.destinations.get_congested_downstream_density(*args),
            self.CE.destinations.get_congested_downstream_density(*map(cs.DM, args))
            .full()
            .squeeze(),
        )


if __name__ == "__main__":
    unittest.main()
