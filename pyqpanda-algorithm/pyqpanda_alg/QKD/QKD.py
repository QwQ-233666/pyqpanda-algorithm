# ~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution
"""
Common infrastructure shared by every quantum-key-distribution protocol.

A QKD protocol lets two parties (Alice and Bob) grow a shared secret key whose
secrecy is guaranteed by quantum mechanics rather than by computational
hardness: the no-cloning theorem and the measurement-disturbance principle mean
that any eavesdropper (Eve) unavoidably imprints detectable errors on the raw
key.  Every protocol here follows the same three stages

    1. quantum transmission   -- Alice encodes random bits into qubits which Bob
                                 measures (implemented with real ``pyqpanda3``
                                 circuits, one circuit per transmitted qubit);
    2. sifting                -- over a public classical channel the two parties
                                 keep only the rounds that are physically
                                 correlated (e.g. matching bases);
    3. parameter estimation   -- a random subset of the sifted bits is disclosed
                                 to estimate the quantum bit-error rate (QBER); a
                                 QBER above the protocol threshold reveals Eve and
                                 the key is discarded.

Sub-classes only implement :meth:`_exchange`, which performs stages 1-2 for a
block of raw qubits and returns Alice's/Bob's raw bits plus the sift mask.  The
base class turns that into a :class:`QKDResult` and exposes :meth:`keygen`.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pyqpanda3.core import CPUQVM, QProg, H, X, RY, CNOT, measure

# measurement / preparation bases
Z_BASIS = 0   # computational basis   |0>, |1>
X_BASIS = 1   # Hadamard basis         |+>, |->

# Measuring "at angle theta" == apply RY(-theta) then measure Z; the X basis is
# theta = pi/2 (RY(-pi/2) sends |+>->|0>, |->->|1>, matching a Hadamard).
_HALF_PI = np.pi / 2.0


class QKDInsecureError(RuntimeError):
    """Raised by :meth:`QKD.keygen` when the estimated QBER exceeds the security
    threshold, i.e. an eavesdropper is suspected and the key must be thrown away."""


@dataclass
class QKDResult:
    """Outcome of a single key-distribution round (a block of ``n_raw`` qubits)."""

    protocol: str
    n_raw: int
    alice_key: str          # Alice's sifted bits (before public disclosure), as '0'/'1'
    bob_key: str            # Bob's sifted bits
    key: str                # final shared secret (sifted key minus the disclosed sample)
    sift_rate: float        # kept fraction = len(sifted) / n_raw
    qber: float             # estimated quantum bit-error rate on the disclosed sample
    secure: bool            # qber <= threshold
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f'QKDResult({self.protocol}, n_raw={self.n_raw}, '
                f'sift_rate={self.sift_rate:.2f}, qber={self.qber:.3f}, '
                f'secure={self.secure}, key_len={len(self.key)})')


class QKD:
    """Base class: quantum transmission helpers, sifting, QBER estimation, keygen.

    Args:
        eavesdropper: if ``True`` an intercept-resend Eve sits on the channel.
        error_rate:   probability that the channel flips a received bit (noise).
        sample_fraction: fraction of the sifted key publicly disclosed to estimate
            the QBER (those bits are removed from the final key).
        qber_threshold: QBER above which the key is deemed insecure (BB84's
            asymptotic bound is ~11 %).
        seed: seed for the (classical) random-number generator, for reproducibility.
    """

    name = 'QKD'
    #: rough kept-fraction after sifting, used only to size the raw block in keygen
    sift_efficiency = 0.5

    def __init__(self, eavesdropper: bool = False, error_rate: float = 0.0,
                 sample_fraction: float = 0.2, qber_threshold: float = 0.11,
                 seed: Optional[int] = None):
        assert 0.0 <= error_rate <= 1.0, 'error_rate must be in [0, 1]'
        assert 0.0 <= sample_fraction < 1.0, 'sample_fraction must be in [0, 1)'
        assert 0.0 <= qber_threshold <= 1.0, 'qber_threshold must be in [0, 1]'
        self.eavesdropper = bool(eavesdropper)
        self.error_rate = float(error_rate)
        self.sample_fraction = float(sample_fraction)
        self.qber_threshold = float(qber_threshold)
        self._rng = np.random.default_rng(seed)
        self._qvm = CPUQVM()

    # ------------------------------------------------------------------ #
    # low-level quantum helpers (one circuit == one transmitted qubit)    #
    # ------------------------------------------------------------------ #
    def _run(self, prog: QProg, n_cbits: int):
        """Run ``prog`` for a single shot and return the measured bits as a list
        indexed by classical-bit number (``cbit j`` at list position ``j``)."""
        self._qvm.run(prog, 1)
        bitstr = next(iter(self._qvm.result().get_counts()))   # big-endian over cbits
        return [int(bitstr[n_cbits - 1 - j]) for j in range(n_cbits)]

    def _pm(self, prep_basis: int, prep_bit: int, meas_basis: int) -> int:
        """Prepare-and-measure primitive: prepare one qubit as ``prep_bit`` in
        ``prep_basis`` (Z/X), measure it in ``meas_basis`` (Z/X), return the outcome."""
        prog = QProg(1)
        if prep_bit:
            prog << X(0)
        if prep_basis == X_BASIS:
            prog << H(0)                 # |0>/|1> -> |+>/|->
        if meas_basis == X_BASIS:
            prog << H(0)                 # rotate X basis back to computational for read-out
        prog << measure(0, 0)
        return self._run(prog, 1)[0]

    def _bell_measure(self, rot0: float, rot1: float):
        """Create a Bell pair |Phi+> = (|00>+|11>)/sqrt2 and measure qubit 0 at
        rotation ``rot0`` and qubit 1 at ``rot1`` (RY angles). Returns (out0, out1).

        For |Phi+> the two spin outcomes correlate as ``<s0 s1> = cos(rot0-rot1)``,
        so equal rotations give identical bits -- the raw key of the entanglement
        protocols."""
        prog = QProg(2)
        prog << H(0) << CNOT(0, 1)
        prog << RY(0, -rot0) << RY(1, -rot1)
        prog << measure(0, 0) << measure(1, 1)
        out = self._run(prog, 2)
        return out[0], out[1]

    def _resend(self, prep_rot: float, prep_bit: int, meas_rot: float) -> int:
        """Prepare the eigenstate that Eve collapsed onto (``prep_bit`` at angle
        ``prep_rot``) and measure it at ``meas_rot``. Used to model intercept-resend
        on an entangled arm."""
        prog = QProg(1)
        if prep_bit:
            prog << X(0)
        prog << RY(0, prep_rot)          # rebuild the measured eigenstate
        prog << RY(0, -meas_rot)         # receiver's measurement rotation
        prog << measure(0, 0)
        return self._run(prog, 1)[0]

    def _maybe_flip(self, bit: int) -> int:
        """Apply the classical channel-noise model to a received bit."""
        if self.error_rate and self._rng.random() < self.error_rate:
            return bit ^ 1
        return bit

    # ------------------------------------------------------------------ #
    # protocol hook + public API                                          #
    # ------------------------------------------------------------------ #
    def _exchange(self, n_raw: int) -> dict:
        """Run quantum transmission + sifting for ``n_raw`` raw qubits.

        Sub-classes must return a dict with keys ``alice`` (int array of Alice's
        raw bits), ``bob`` (Bob's raw bits), ``keep`` (bool sift mask), and an
        optional ``extra`` dict of protocol-specific diagnostics."""
        raise NotImplementedError

    def distribute(self, n_raw: int) -> QKDResult:
        """Distribute a block of ``n_raw`` raw qubits and return a :class:`QKDResult`."""
        n_raw = int(n_raw)
        assert n_raw >= 1, 'n_raw must be >= 1'
        data = self._exchange(n_raw)

        alice = np.asarray(data['alice'], dtype=int)
        bob = np.asarray(data['bob'], dtype=int)
        keep = np.asarray(data['keep'], dtype=bool)

        a_sift = alice[keep]
        b_sift = bob[keep]
        n_sift = int(a_sift.size)
        sift_rate = n_sift / n_raw

        # publicly disclose a random sample to estimate the QBER
        n_sample = int(round(self.sample_fraction * n_sift))
        if (self.eavesdropper or self.error_rate > 0.0) and n_sift > 0:
            n_sample = max(1, n_sample)
        if n_sift > 1:
            n_sample = min(n_sample, n_sift - 1)   # never disclose the whole key

        if n_sample > 0:
            idx = self._rng.choice(n_sift, size=n_sample, replace=False)
            qber = float(np.mean(a_sift[idx] != b_sift[idx]))
            mask = np.ones(n_sift, dtype=bool)
            mask[idx] = False
            key_bits = a_sift[mask]
        else:
            qber = float(np.mean(a_sift != b_sift)) if n_sift else 0.0
            key_bits = a_sift

        return QKDResult(
            protocol=self.name,
            n_raw=n_raw,
            alice_key=_bits_to_str(a_sift),
            bob_key=_bits_to_str(b_sift),
            key=_bits_to_str(key_bits),
            sift_rate=sift_rate,
            qber=qber,
            secure=bool(qber <= self.qber_threshold),
            extra=dict(data.get('extra', {})),
        )

    def keygen(self, nlen: int) -> str:
        """Generate a shared secret key of ``nlen`` bits (returned as a '0'/'1' string).

        Raw qubits are transmitted in blocks until enough sifted bits accumulate.
        If an eavesdropper is detected (QBER over threshold) a
        :class:`QKDInsecureError` is raised and the key is discarded."""
        assert int(nlen) >= 1, 'nlen must be >= 1'
        nlen = int(nlen)

        # size a raw block from the expected sift efficiency and disclosure loss
        yield_per_raw = max(self.sift_efficiency * (1.0 - self.sample_fraction), 1e-3)
        block = int(nlen / yield_per_raw) + 32

        key = ''
        while len(key) < nlen:
            res = self.distribute(block)
            if not res.secure:
                raise QKDInsecureError(
                    f'{self.name}: estimated QBER {res.qber:.3f} exceeds threshold '
                    f'{self.qber_threshold:.3f} -- eavesdropping suspected, key discarded')
            key += res.key
            block = max(32, block // 2)   # top-ups can be smaller
        return key[:nlen]


def _bits_to_str(bits) -> str:
    return ''.join('1' if b else '0' for b in np.asarray(bits, dtype=int))
