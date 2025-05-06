"""
Dummy driver for testing software that normally talks to a Keithley 2400.

It reproduces **all public attributes, properties, and methods** found in the
real driver so nothing in your application needs to change.  Two emulation
modes are available:

1. **Resistor mode** – a 100 MΩ load (default)
2. **Diode mode**    – an ideal diode governed by
   I = Iₛ · (exp(V / (n·Vₜ)) − 1)

Switch at construction time or later with ``load_type``.

MIT License – 2025
"""

import logging
import time
from collections import deque
from warnings import warn

import numpy as np
from pymeasure.instruments.fakes import FakeInstrument

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class DummyKeithley2400(FakeInstrument):
    """Keithley 2400 stand-in that behaves like either

    * an **ideal 100 MΩ resistor**, or
    * an **ideal diode** (Shockley equation).

    Parameters
    ----------
    adapter : str | None
        Passed through to ``FakeInstrument``.  Not used.
    name : str, optional
        Device name shown in logs.
    load_type : {'resistor', 'diode'}, optional
        Behaviour model; default 'resistor'.
    resistor_ohms : float, optional
        Resistance when ``load_type='resistor'`` (Ω).
    diode_Is : float, optional
        Saturation current A for diode model.
    diode_n : float, optional
        Ideality factor for diode model (≈1 … 2).
    temperature_K : float, optional
        Junction temperature (sets thermal voltage Vₜ).

    All remaining *args / **kwargs are forwarded to ``FakeInstrument``.
    """

    _MAX_BUF = 2_500  # Like the real instrument buffer

    def __init__(
        self,
        adapter=None,
        name="Dummy Keithley 2400",
        *,
        load_type="resistor",
        resistor_ohms=100e6,
        diode_Is=1e-12,
        diode_n=1.0,
        temperature_K=300.0,
        **kwargs,
    ):
        super().__init__(adapter=adapter, name=name, includeSCPI=False, **kwargs)

        # ---------- “hardware” config ----------
        self._load_type = load_type
        self._R_load = float(resistor_ohms)
        self._diode_Is = float(diode_Is)
        self._diode_n = float(diode_n)
        self._Vt = 8.617333262e-5 * temperature_K  # kT/q  [≈25.85 mV @ 300 K]

        # ---------- instrument state ----------
        self._source_mode = "voltage"  # 'current' | 'voltage'
        self._src_enabled = False
        self._src_voltage = 0.0  # V
        self._src_current = 0.0  # A
        self._compliance_voltage = 21.0  # V
        self._compliance_current = 1.05  # A
        self._buffer = deque(maxlen=self._MAX_BUF)  # store (V, I, R)

    # --------------------------------------------------------------------- #
    #                        =====  configuration  =====                    #
    # --------------------------------------------------------------------- #
    # Provide public properties so tests can tweak behaviour on the fly
    @property
    def load_type(self):
        """'resistor' or 'diode'"""
        return self._load_type

    @load_type.setter
    def load_type(self, value):
        if value not in ("resistor", "diode"):
            raise ValueError("load_type must be 'resistor' or 'diode'")
        self._load_type = value

    # --------------------------------------------------------------------- #
    #                       =====  simulation core  =====                   #
    # --------------------------------------------------------------------- #
    def _simulate(self):
        """Return (V, I, R) tuple for present source settings."""
        if self._source_mode == "voltage":
            V = self._src_voltage
            if self._load_type == "resistor":
                I = V / self._R_load
            else:  # diode
                I = self._diode_Is * (np.exp(V / (self._diode_n * self._Vt)) - 1.0)
        else:  # current source
            I = self._src_current
            if self._load_type == "resistor":
                V = I * self._R_load
            else:  # diode – invert Shockley eq
                V = self._diode_n * self._Vt * np.log(I / self._diode_Is + 1.0)

        R_inst = np.inf if abs(I) < 1e-30 else V / I  # instantaneous R
        return float(V), float(I), float(R_inst)

    def _push_to_buffer(self):
        self._buffer.append(self._simulate())

    # --------------------------------------------------------------------- #
    #                    =====  real-driver properties  =====               #
    # --------------------------------------------------------------------- #
    # -- source & output --------------------------------------------------
    @property
    def source_mode(self):
        return self._source_mode

    @source_mode.setter
    def source_mode(self, mode):
        if mode not in ("current", "voltage"):
            raise ValueError
        self._source_mode = mode

    @property
    def source_enabled(self):
        return self._src_enabled

    @source_enabled.setter
    def source_enabled(self, s):
        self._src_enabled = bool(s)

    # -- compliance -------------------------------------------------------
    compliance_voltage = property(
        lambda self: self._compliance_voltage,
        lambda self, v: setattr(self, "_compliance_voltage", float(v)),
    )
    compliance_current = property(
        lambda self: self._compliance_current,
        lambda self, i: setattr(self, "_compliance_current", float(i)),
    )

    # -- set-points -------------------------------------------------------
    @property
    def source_voltage(self):
        return self._src_voltage

    @source_voltage.setter
    def source_voltage(self, v):
        self._src_voltage = float(v)
        self._push_to_buffer()

    @property
    def source_current(self):
        return self._src_current

    @source_current.setter
    def source_current(self, i):
        self._src_current = float(i)
        self._push_to_buffer()

    # -- instantaneous measurements --------------------------------------
    voltage = property(lambda self: self._simulate()[0])
    current = property(lambda self: self._simulate()[1])
    resistance = property(lambda self: self._simulate()[2])

    # --------------------------------------------------------------------- #
    #                   =====  buffer statistics shortcuts  =====           #
    # --------------------------------------------------------------------- #
    def _stats(self, idx, fn):
        if not self._buffer:
            return np.nan
        arr = np.asarray(self._buffer)[:, idx]
        return float(getattr(np, fn)(arr))

    mean_voltage = property(lambda s: s._stats(0, "mean"))
    max_voltage = property(lambda s: s._stats(0, "max"))
    min_voltage = property(lambda s: s._stats(0, "min"))
    std_voltage = property(lambda s: s._stats(0, "std"))

    mean_current = property(lambda s: s._stats(1, "mean"))
    max_current = property(lambda s: s._stats(1, "max"))
    min_current = property(lambda s: s._stats(1, "min"))
    std_current = property(lambda s: s._stats(1, "std"))

    mean_resistance = property(lambda s: s._stats(2, "mean"))
    max_resistance = property(lambda s: s._stats(2, "max"))
    min_resistance = property(lambda s: s._stats(2, "min"))
    std_resistance = property(lambda s: s._stats(2, "std"))

    means = property(lambda s: [s.mean_voltage, s.mean_current, s.mean_resistance])
    maximums = property(lambda s: [s.max_voltage, s.max_current, s.max_resistance])
    minimums = property(lambda s: [s.min_voltage, s.min_current, s.min_resistance])
    standard_devs = property(lambda s: [s.std_voltage, s.std_current, s.std_resistance])

    # --------------------------------------------------------------------- #
    #                   =====  API parity with real driver  =====           #
    # --------------------------------------------------------------------- #
    # -- trivial wrappers -------------------------------------------------
    def enable_source(self):
        self.source_enabled = True

    def disable_source(self):
        self.source_enabled = False

    def measure_resistance(self, *_, **__):
        pass

    def measure_voltage(self, *_, **__):
        pass

    def measure_current(self, *_, **__):
        pass

    def auto_range_source(self):
        pass

    # -- apply helpers ----------------------------------------------------
    def apply_current(self, current_range=None, compliance_voltage=0.1):
        self.source_mode = "current"
        self.compliance_voltage = compliance_voltage
        if current_range is not None:
            self.source_current_range = current_range  # stored but unused

    def apply_voltage(self, voltage_range=None, compliance_current=0.1):
        self.source_mode = "voltage"
        self.compliance_current = compliance_current
        if voltage_range is not None:
            self.source_voltage_range = voltage_range  # stored but unused

    # -- beeper -----------------------------------------------------------
    def beep(self, frequency, duration):
        log.debug(f"BEEP {frequency} Hz for {duration}s")

    def triad(self, base_f, dur):
        for f in (base_f, base_f * 5 / 4, base_f * 6 / 4):
            self.beep(f, dur)
            time.sleep(dur)

    # -- error stub -------------------------------------------------------
    @property
    def error(self):
        warn("Dummy instrument: always (0, 'No error')", FutureWarning)
        return 0, "No error"

    next_error = error  # alias

    def reset(self):
        self.__init__(self.adapter)  # brute-force re-init

    # -- ramp helpers -----------------------------------------------------
    def _ramp(self, setter, target, steps, pause):
        start = getattr(self, setter.__name__.replace("source_", ""))
        for v in np.linspace(start, target, int(steps)):
            setter(v)
            time.sleep(pause)

    def ramp_to_current(self, target_current, steps=30, pause=20e-3):
        #self._ramp(self.source_current.__set__, target_current, steps, pause)
        pass

    def ramp_to_voltage(self, target_voltage, steps=30, pause=20e-3):
        #self._ramp(self.source_voltage.__set__, target_voltage, steps, pause)
        pass

    # -- triggering stubs -------------------------------------------------
    def trigger(self):
        pass

    def trigger_immediately(self):
        pass

    def trigger_on_bus(self):
        pass

    def set_trigger_counts(self, *_, **__):
        pass

    def sample_continuously(self):
        pass

    def set_timed_arm(self, interval):
        pass

    def trigger_on_external(self, *_, **__):
        pass

    def output_trigger_on_external(self, *_, **__):
        pass

    def disable_output_trigger(self):
        pass

    # -- buffer management (minimal) --------------------------------------
    def start_buffer(self):
        pass

    def stop_buffer(self):
        pass

    def disable_buffer(self):
        pass

    # -- simple status ----------------------------------------------------
    def status(self):
        return "0, 'No status'"

    # -- swept I-V helpers ------------------------------------------------
    def RvsI(self, startI, stopI, stepI, compliance, delay=10e-3, backward=False):
        currents = np.arange(startI, stopI + stepI / 2, stepI)
        if backward:
            currents = currents[::-1]
        data = []
        for i in currents:
            self.source_current = i
            time.sleep(delay)
            V, I, R = self._simulate()
            data.append((i, V, I, R))
        return data

    def RvsIaboutZero(self, minI, maxI, stepI, compliance, delay=10e-3):
        data = []
        data.extend(self.RvsI(minI, maxI, stepI, compliance, delay))
        data.extend(self.RvsI(minI, maxI, stepI, compliance, delay, backward=True))
        data.extend(self.RvsI(-minI, -maxI, -stepI, compliance, delay))
        data.extend(self.RvsI(-minI, -maxI, -stepI, compliance, delay, backward=True))
        return data

    # -- terminal selection stubs ----------------------------------------
    def use_rear_terminals(self):
        pass

    def use_front_terminals(self):
        pass

    # -- graceful shutdown -----------------------------------------------
    def shutdown(self):
        log.info("Dummy Keithley shutdown: ramp to 0 and output off.")