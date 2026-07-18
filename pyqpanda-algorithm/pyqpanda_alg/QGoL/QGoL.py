# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quantum implementation of one-dimensional (elementary) cellular automata.

A one-dimensional elementary cellular automaton (ECA) updates every cell of a
1D lattice from the states of itself and its two nearest neighbours
``(left, center, right)`` according to a fixed rule.  Wolfram encodes such a
rule as an 8-bit integer, so there are 256 possible rules; ``Rule 110`` is one
of the most famous ones (it is Turing-complete).

This module implements the rule evaluation as a *reversible* quantum circuit.
Because an ECA rule is a Boolean function of three bits, it can be written in
**algebraic normal form (ANF)** -- an XOR of AND terms:

    f(L, C, R) = a0
               ⊕ aL·L ⊕ aC·C ⊕ aR·R
               ⊕ aLC·L·C ⊕ aLR·L·R ⊕ aCR·C·R
               ⊕ aLCR·L·C·R

Each AND term is evaluated with a Toffoli / multi-controlled-X gate into a
scratch qubit, XOR-ed (CNOT) into the target, and then uncomputed, so the whole
update is fully reversible and preserves the original lattice.

Two usage modes are provided:

* ``evolve``  -- the scalable mode.  Each generation is produced by running a
  fresh quantum circuit for the rule, reading the result, and re-preparing the
  next input.  Qubit cost is ``O(n)`` and independent of the number of steps.
* ``build_spacetime_circuit`` -- the "fully quantum" mode.  A single circuit
  stores *every* generation in its own register, producing the whole spacetime
  diagram from one run.  Qubit cost is ``O(n · steps)`` and is therefore only
  suitable for very small instances (it is guarded by ``max_qubits``).

Examples
--------
>>> from pyqpanda_alg.QGoL import Rule110
>>> ca = Rule110(n=11)
>>> # single '1' in the middle, evolve 15 generations and print the diagram
>>> gens = ca.evolve(initial=[0]*5 + [1] + [0]*5, steps=15)
>>> print(ca.diagram(gens))
"""

from typing import Optional

from pyqpanda3.core import CPUQVM, QCircuit, QProg, X, CNOT, TOFFOLI


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------
def _rule_to_anf(rule: int) -> dict[str, int]:
    """
    Convert a Wolfram elementary-cellular-automaton rule (0..255) into the
    coefficients of its algebraic normal form (XOR-of-ANDs) over the three
    inputs ``(L, C, R)``.

    The rule's 8-bit truth table assigns, to every neighbourhood
    ``b = 4·L + 2·C + R``, the output bit ``(rule >> b) & 1``.  The ANF
    coefficients are obtained by a Möbius (Walsh-Hadamard) transform.

    Parameters
        rule : ``int``
            An integer in ``[0, 255]`` identifying the elementary CA rule.

    Returns
        anf : ``dict[str, int]``
            Dictionary with keys ``a0, aL, aC, aR, aLC, aLR, aCR, aLCR``.  A key
            value of ``1`` means the corresponding AND term participates in the
            rule.

    Examples
        Rule 110 has ANF ``C ⊕ R ⊕ (C·R) ⊕ (L·C·R)``::

            >>> anf = _rule_to_anf(110)
            >>> (anf['aC'], anf['aR'], anf['aCR'], anf['aLCR'])
            (1, 1, 1, 1)
    """
    if not (0 <= rule <= 255):
        raise ValueError("rule must be an integer in [0, 255]")

    # f[(L << 2) | (C << 1) | R] = output bit
    f = [(rule >> b) & 1 for b in range(8)]

    def F(L: int, C: int, R: int) -> int:
        return f[(L << 2) | (C << 1) | R]

    anf = {
        "a0":   F(0, 0, 0),
        "aL":   F(0, 0, 0) ^ F(1, 0, 0),
        "aC":   F(0, 0, 0) ^ F(0, 1, 0),
        "aR":   F(0, 0, 0) ^ F(0, 0, 1),
        "aLC":  F(0, 0, 0) ^ F(1, 0, 0) ^ F(0, 1, 0) ^ F(1, 1, 0),
        "aLR":  F(0, 0, 0) ^ F(1, 0, 0) ^ F(0, 0, 1) ^ F(1, 0, 1),
        "aCR":  F(0, 0, 0) ^ F(0, 1, 0) ^ F(0, 0, 1) ^ F(0, 1, 1),
    }
    # cubic coefficient is the parity of all eight truth-table entries
    aLCR = 0
    for L in (0, 1):
        for C in (0, 1):
            for R in (0, 1):
                aLCR ^= F(L, C, R)
    anf["aLCR"] = aLCR
    return anf


# ---------------------------------------------------------------------------
# Quantum elementary cellular automaton
# ---------------------------------------------------------------------------
class Quantum1DCellularAutomaton:
    """
    Quantum (reversible) simulation of a one-dimensional elementary cellular
    automaton for an arbitrary Wolfram rule (defaulting to Rule 110).

    The next state of every cell is computed reversibly with Toffoli /
    multi-controlled-X gates, using the rule's algebraic normal form, into a
    fresh register while keeping the original lattice unchanged.

    Parameters
        rule : ``int``, optional
            Wolfram rule number in ``[0, 255]``. Default ``110``.
        n : ``Optional[int]``, optional
            Number of cells.  If ``None`` it is inferred from the first state
            passed to :meth:`step` / :meth:`evolve`.
        boundary : ``str``, optional
            Boundary condition, either ``'periodic'`` (torus, default) or
            ``'fixed'`` (missing neighbours are treated as ``0``).

    Examples
        A single quantum step of Rule 110 on a 5-cell lattice::

            >>> ca = QuantumElementaryCA(rule=110, n=5)
            >>> ca.step([1, 0, 1, 1, 0])
            [1, 1, 1, 0, 1]
    """

    def __init__(
        self,
        rule: int = 110,
        n: Optional[int] = None,
        boundary: str = "periodic",
    ) -> None:
        if boundary not in ("periodic", "fixed"):
            raise ValueError("boundary must be 'periodic' or 'fixed'")
        self.rule = rule
        self.n = n
        self.boundary = boundary
        self._anf = _rule_to_anf(rule)

    # -- low level: reversible rule evaluation for one cell -----------------
    def _cell_circuit(
        self,
        target,
        qL,
        qC,
        qR,
        sq_quad,
        sq_cubic,
        anf: dict[str, int],
    ) -> QCircuit:
        """
        Build a reversible sub-circuit that computes

            target := target ⊕ f(L, C, R)

        into ``target`` (which must start in the ``|0>`` state) using the
        supplied ANF coefficients.  The scratch qubits ``sq_quad`` / ``sq_cubic``
        are borrowed and returned to ``|0>`` so the computation is fully
        reversible.  A neighbour passed as ``None`` is treated as the constant
        ``0`` (used for 'fixed' boundaries), which automatically omits every
        term that depends on it.
        """
        cir = QCircuit()

        # constant term
        if anf["a0"]:
            cir << X(target)

        # linear terms
        if anf["aL"] and qL is not None:
            cir << CNOT(qL, target)
        if anf["aC"]:
            cir << CNOT(qC, target)
        if anf["aR"] and qR is not None:
            cir << CNOT(qR, target)

        # quadratic terms (each evaluated into sq_quad, XOR-ed, then uncomputed)
        if anf["aLC"] and qL is not None:
            cir << TOFFOLI(qL, qC, sq_quad) \
                << CNOT(sq_quad, target) \
                << TOFFOLI(qL, qC, sq_quad)
        if anf["aLR"] and qL is not None and qR is not None:
            cir << TOFFOLI(qL, qR, sq_quad) \
                << CNOT(sq_quad, target) \
                << TOFFOLI(qL, qR, sq_quad)
        if anf["aCR"] and qR is not None:
            cir << TOFFOLI(qC, qR, sq_quad) \
                << CNOT(sq_quad, target) \
                << TOFFOLI(qC, qR, sq_quad)

        # cubic term (3-controlled-X into sq_cubic, XOR-ed, then uncomputed)
        if anf["aLCR"] and qL is not None and qR is not None:
            cir << X(sq_cubic).control([qL, qC, qR]) \
                << CNOT(sq_cubic, target) \
                << X(sq_cubic).control([qL, qC, qR])

        return cir

    def _neighbours(self, reg, i: int) -> tuple:
        """Return ``(qL, qC, qR)`` for cell ``i`` according to the boundary."""
        n = len(reg)
        qC = reg[i]
        if self.boundary == "periodic":
            qL = reg[(i - 1) % n]
            qR = reg[(i + 1) % n]
        else:  # fixed boundary, missing neighbours are None (treated as 0)
            qL = reg[i - 1] if i > 0 else None
            qR = reg[i + 1] if i < n - 1 else None
        return qL, qC, qR

    # -- build a quantum circuit for a single transition --------------------
    def build_step_circuit(self, current: list[int]):
        """
        Construct the *canonical* reversible circuit for one full transition:
        it encodes ``current`` into a ``state`` register and computes the whole
        next generation into a separate ``next`` register in a single run,
        keeping the original lattice unchanged.  This needs ``2n + 2`` qubits,
        so it is best used for inspection / small ``n``.  (:meth:`step` uses a
        lighter ``n + 3`` qubit variant internally.)

        Parameters
            current : ``list[int]``
                Classical binary configuration of length ``n`` (``0``/``1``).

        Returns
            prog : ``QProg``
                The quantum program (un-run).
            machine : ``CPUQVM``
                The simulator instance holding the allocated qubits.  The caller
                runs ``machine.run(prog, 1000)`` and reads out via
                ``machine.result().get_prob_dict(...)``.
            qubits : ``dict``
                Handles ``{'state', 'next', 'sq_quad', 'sq_cubic'}`` for
                inspection / drawing.
        """
        if self.n is None:
            self.n = len(current)
        if len(current) != self.n:
            raise ValueError(f"length of current ({len(current)}) != n ({self.n})")

        machine = CPUQVM()
        total = 2 * self.n + 2
        prog = QProg(total)
        q = prog.qubits()
        state = q[:self.n]
        nxt = q[self.n:2 * self.n]
        sq_quad = q[2 * self.n]      # scratch for quadratic AND terms
        sq_cubic = q[2 * self.n + 1] # scratch for the cubic AND term

        # encode the classical input into the state register
        for i, bit in enumerate(current):
            if bit:
                prog << X(state[i])

        # compute the next generation cell by cell
        for i in range(self.n):
            qL, qC, qR = self._neighbours(state, i)
            prog << self._cell_circuit(nxt[i], qL, qC, qR, sq_quad, sq_cubic, self._anf)

        return prog, machine, {
            "state": state,
            "next": nxt,
            "sq_quad": sq_quad,
            "sq_cubic": sq_cubic,
        }

    # -- run one quantum step and read the result ---------------------------
    def step(self, current: list[int]) -> list[int]:
        """
        Execute a single quantum transition and return the next generation as a
        classical bit list.

        The next state of each cell is computed with the *reversible* rule
        circuit (:meth:`_cell_circuit`) into a single ancilla qubit, then read
        out.  Because the input lattice is a classical computational-basis
        state, the ancilla ends up in a definite ``|0>`` / ``|1>`` state, so a
        single run per cell suffices.  The whole step only needs ``n + 3``
        qubits (``n`` lattice qubits + 1 output ancilla + 2 scratch qubits),
        which keeps the statevector simulator tractable for modest ``n``.

        Parameters
            current : ``list[int]``
                Current generation (binary, length ``n``).

        Returns
            next_gen : ``list[int]``
                Next generation computed by the (quantum, reversible) rule.
        """
        if self.n is None:
            self.n = len(current)
        if len(current) != self.n:
            raise ValueError(f"length of current ({len(current)}) != n ({self.n})")

        n = self.n
        anf = self._anf
        next_gen = [0] * n

        for i in range(n):
            # A fresh machine + fresh QProg per cell keeps the lattice at |0>
            # and bounds the qubit pool (only n + 3 qubits are used).
            machine = CPUQVM()
            prog = QProg(n + 3)
            q = prog.qubits()
            state = q[:n]
            anc = q[n]                # holds f(neighbours of cell i)
            anc_q = q[n:n + 1]  # same qubit, as a QVec for measurement
            sq_quad = q[n + 1]        # scratch for quadratic AND terms
            sq_cubic = q[n + 2]       # scratch for the cubic AND term

            # encode the classical input lattice
            for j, bit in enumerate[int](current):
                if bit:
                    prog << X(state[j])
            # compute the next state of cell i reversibly into `anc`
            qL, qC, qR = self._neighbours(state, i)
            prog << self._cell_circuit(anc, qL, qC, qR, sq_quad, sq_cubic, anf)

            machine.run(prog, 1000)
            prob = machine.result().get_prob_dict(anc_q)
            next_gen[i] = 1 if prob.get("1", 0.0) > 0.5 else 0

        return next_gen

    # -- evolve many generations (scalable, re-prepare mode) ----------------
    def evolve(self, initial: list[int], steps: int, verbose: bool = False) -> list[list[int]]:
        """
        Evolve the automaton for ``steps`` generations.

        Each generation is produced by running a fresh quantum circuit for the
        rule (the reversible update), reading the outcome, and re-preparing it
        as the input of the next step.  This keeps the qubit cost at ``O(n)``
        regardless of ``steps``.

        Parameters
            initial : ``list[int]``
                Initial generation (binary, length ``n``).
            steps : ``int``
                Number of generations to advance (``gen_0`` is included in the
                returned list, so ``len(result) == steps + 1``).
            verbose : ``bool``, optional
                If ``True`` print the evolving diagram line by line.

        Returns
            generations : ``list[list[int]]``
                ``generations[t]`` is the configuration at time ``t``.
        """
        generations = [list(initial)]
        current = list(initial)
        for _ in range(steps):
            current = self.step(current)
            generations.append(current)
            if verbose:
                print("".join("#" if b else " " for b in current))
        return generations

    # -- fully-quantum spacetime circuit (history registers) ----------------
    def build_spacetime_circuit(self, initial: list[int], steps: int, max_qubits: int = 18):
        """
        Build a *single* quantum circuit that computes **all** ``steps + 1``
        generations at once, storing every generation in its own register.

        This is the genuinely "in-place quantum evolution": no classical
        re-preparation is used.  Because each generation occupies ``n`` qubits,
        the total qubit count is ``(steps + 1) · n + 2``.  The builder raises
        ``ValueError`` when this exceeds ``max_qubits`` (a statevector simulator
        cannot handle that many qubits).

        Parameters
            initial : ``list[int]``
                Initial generation (binary, length ``n``).
            steps : ``int``
                Number of transitions to embed.
            max_qubits : ``int``, optional
                Safety cap on the number of allocated qubits.

        Returns
            prog : ``QProg``
            machine : ``CPUQVM``
            registers : ``list``
                ``registers[t]`` holds generation ``t``; useful for measuring
                the full spacetime diagram in one run.
        """
        if self.n is None:
            self.n = len(initial)
        n = self.n
        total_qubits = (steps + 1) * n + 2
        if total_qubits > max_qubits:
            raise ValueError(
                f"spacetime circuit needs {total_qubits} qubits "
                f"(> max_qubits={max_qubits}); use 'evolve' for larger instances"
            )

        machine = CPUQVM()
        total = (steps + 1) * n + 2
        prog = QProg(total)
        q = prog.qubits()
        registers = [q[t * n:(t + 1) * n] for t in range(steps + 1)]
        sq_quad = q[(steps + 1) * n]      # scratch for quadratic AND terms
        sq_cubic = q[(steps + 1) * n + 1] # scratch for the cubic AND term

        for i, bit in enumerate(initial):
            if bit:
                prog << X(registers[0][i])

        for t in range(steps):
            cur = registers[t]
            nxt = registers[t + 1]
            for i in range(n):
                qL, qC, qR = self._neighbours(cur, i)
                prog << self._cell_circuit(nxt[i], qL, qC, qR, sq_quad, sq_cubic, self._anf)

        return prog, machine, registers

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def diagram(generations: list[list[int]], live: str = "#", dead: str = " ") -> str:
        """
        Render a list of generations as an ASCII spacetime diagram.

        Parameters
            generations : ``list[list[int]]``
                Output of :meth:`evolve` (or manually supplied).
            live : ``str``, optional
                Character for a live cell.
            dead : ``str``, optional
                Character for a dead cell.

        Returns
            diagram : ``str``
                Multi-line string, one row per generation.
        """
        return "\n".join("".join(live if b else dead for b in gen) for gen in generations)

    def verify(self, initial: list[int], steps: int) -> bool:
        """
        Cross-check the quantum evolution against a classical evaluation of the
        rule, returning ``True`` if they match for all generations.

        The classical reference also uses the ANF but evaluates it directly on
        bits, providing an independent confirmation of the quantum circuit.
        """
        quantum = self.evolve(initial, steps)
        assert self.n is not None
        classical = [list(initial)]
        cur = list(initial)
        for _ in range(steps):
            nxt = [0] * self.n
            for i in range(self.n):
                qL, qC, qR = self._neighbours(cur, i)
                nxt[i] = self._anf_eval(qL, qC, qR)
            cur = nxt
            classical.append(cur)
        return quantum == classical

    def _anf_eval(self, L, C, R) -> int:
        lft = 0 if L is None else L
        rgt = 0 if R is None else R
        a = self._anf
        val = a["a0"]
        val ^= a["aL"] & lft
        val ^= a["aC"] & C
        val ^= a["aR"] & rgt
        val ^= a["aLC"] & (lft & C)
        val ^= a["aLR"] & (lft & rgt)
        val ^= a["aCR"] & (C & rgt)
        val ^= a["aLCR"] & (lft & C & rgt)
        return val & 1


class Rule110(Quantum1DCellularAutomaton):
    """
    Convenience class for the quantum implementation of **Rule 110**, the
    Turing-complete elementary cellular automaton.

    Examples
        Build the classic Rule-110 triangle from a single live cell::

            >>> ca = Rule110(n=11)
            >>> gens = ca.evolve([0]*5 + [1] + [0]*5, steps=10)
            >>> print(ca.diagram(gens[:4]))
    """

    def __init__(self, n: int = None, boundary: str = "periodic") -> None:
        super().__init__(rule=110, n=n, boundary=boundary)


if __name__ == "__main__":
    # Demo: classic Rule 110 spacetime diagram from a single live cell.
    # NOTE: a statevector simulator can only handle a modest number of qubits,
    # so `n` is kept small here (the reversible step uses n + 3 qubits).
    n = 7
    #initial = [0] * (n // 2) + [1] + [0] * (n // 2)
    initial = [0] * (n - 1) + [1]
    ca = Rule110(n=n)
    generations = ca.evolve(initial, steps=30)
    print(f"Rule {ca.rule} ({ca.boundary} boundary), {n} cells, {len(generations)} generations\n")
    print(ca.diagram(generations))
    print("\nQuantum vs classical consistency:", ca.verify(initial, steps=30))
