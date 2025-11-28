import os  # Import the os module
import sys
import time  # Import the time module
from time import sleep

# -- Force a non‐GUI backend before importing pyplot --
import matplotlib
import numpy as np
from DevicePicker import DevicePicker
from pymeasure.adapters import PrologixAdapter, VISAAdapter
from PyQt5.QtGui import QIcon  # Add this import

from drivers.dummy_keithley import DummyKeithley2400

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # For creating and saving your own plots
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows.managed_dock_window import ManagedDockWindow
from pymeasure.experiment import (
    BooleanParameter,
    FloatParameter,
    IntegerParameter,
    ListParameter,
    Metadata,
    Parameter,
    Procedure,
)

# Both the 6430 and 2450 use essentially the same commands, so the 2400 driver works fine
from pymeasure.instruments.keithley import Keithley2400

import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

DEFAULT_GPIB_ADDRESS = 5
try:
    DEFAULT_GPIB_ADDRESS = int(
        os.environ.get("KEITHLEYCONTROL_GPIB_ADDRESS", DEFAULT_GPIB_ADDRESS)
    )
except ValueError:
    pass


def _consume_test_flag(argv):
    """Return True if -TEST is present; strip it so Qt won't see it."""
    test_tokens = {"-TEST", "--TEST", "/TEST"}
    found = False
    keep = []
    for arg in argv:
        if arg.upper() in test_tokens:
            found = True
            continue
        keep.append(arg)
    argv[:] = keep
    return found


ALLOW_DUMMY = _consume_test_flag(sys.argv)


def _dialog_exec(dialog):
    exec_fn = getattr(dialog, "exec", None) or getattr(dialog, "exec_", None)
    return exec_fn()


def _looks_like_serial_resource(resource: str) -> bool:
    if not resource:
        return False
    text = resource.strip().lower()
    return (
        text.startswith("asrl")
        or text.startswith("com")
        or text.startswith("/dev/")
        or text.startswith("tty")
    )


def _normalize_serial_resource(resource: str) -> str:
    res = resource.strip()
    upper = res.upper()
    if upper.startswith("ASRL"):
        return res if upper.endswith("::INSTR") else f"{res}::INSTR"
    if upper.startswith("COM"):
        digits = "".join(ch for ch in res if ch.isdigit())
        return f"ASRL{digits}::INSTR" if digits else res
    if res.startswith("/"):
        return f"ASRL{res}::INSTR"
    return res


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in PyInstaller bundle, use script's directory
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)


class JVJTProcedure(Procedure):
    device_selection = None
    preconnected = None
    test_mode = ALLOW_DUMMY
    measurement_mode = ListParameter(
        "Measurement mode",
        ["JV", "JT"],
        default="JV",
    )
    identifier = Parameter("Identifier", default="Device")
    # NPLC settings
    max_speed = BooleanParameter("Maximize Measurement Speed", default=False)
    nplc_val = FloatParameter(
        "NPLC value", default=10, group_by="max_speed", group_condition=False
    )
    # JV Params
    minimum_voltage = FloatParameter(
        "Minimum Voltage",
        units="V",
        default=-10,
        group_by="measurement_mode",
        group_condition="JV",
    )
    maximum_voltage = FloatParameter(
        "Maximum Voltage",
        units="V",
        default=10,
        group_by="measurement_mode",
        group_condition="JV",
    )
    step_size = FloatParameter(
        "Step Size",
        units="V",
        default=1,
        group_by="measurement_mode",
        group_condition="JV",
    )
    sweep_speed = FloatParameter(
        "Sweep Speed",
        units="V/s",
        default=1,
        group_by="measurement_mode",
        group_condition="JV",
    )
    sweep_mode = ListParameter(
        "Sweep Type",
        ["Standard Sweep", "Zero-Centered"],
        default="Standard Sweep",
        group_by="measurement_mode",
        group_condition="JV",
    )

    # JT Params
    hold_voltage = FloatParameter(
        "Hold Voltage",
        units="V",
        default=1,
        group_by="measurement_mode",
        group_condition="JT",
    )
    indefinite_measurement = BooleanParameter(
        "Indefinite Measurement",
        default=False,
        group_by="measurement_mode",
        group_condition="JT",
    )
    measurement_time = IntegerParameter(
        "Measurement Duration",
        units="S",
        default=30,
        group_by={"measurement_mode": "JT", "indefinite_measurement": False},
    )
    measurement_interval = FloatParameter(
        "Measurement Interval",
        units="S",
        default=1,
        group_by="measurement_mode",
        group_condition="JT",
    )

    DATA_COLUMNS = [
        "Current JV (A)",
        "Voltage JV (V)",
        "Time JV (S)",
        "Current JT (A)",
        "Voltage JT (V)",
        "Time JT (S)",
    ]

    sm_type_metadata = Metadata("Sourcemeter Type", default="None")
    test_time_metadata = Metadata("Test Time", default="None")

    def _connect_dummy(self):
        self.sourcemeter = DummyKeithley2400()
        self.Sourcemeter_type = "Dummy"
        log.info("Using dummy sourcemeter (test mode).")

    def _configure_2450_compat(self):
        try:
            self.sourcemeter.write("SYST:LANG SCPI")
            self.sourcemeter.write("SYST:CONS 2400")
            log.info("Configured 2450 for 2400 SCPI compatibility.")
        except Exception as exc:
            log.warning(f"Failed to set 2450 compatibility: {exc}")

    def _detect_model(self):
        try:
            ident = self.sourcemeter.ask("*IDN?")
            if ident:
                log.info(f"Instrument ID: {ident.strip()}")
                for model in ("2450", "2400", "6430"):
                    if model in ident:
                        return model
        except Exception as exc:
            log.warning(f"Could not query instrument ID: {exc}")
        return None

    def _connect_instrument(self):
        pre = getattr(JVJTProcedure, "preconnected", None)
        if pre and pre.get("sourcemeter"):
            self.sourcemeter = pre["sourcemeter"]
            self.adapter = pre.get("adapter")
            sel = pre.get("selection") or {}
            if not getattr(self, "device_selection", None):
                self.device_selection = sel
            self.Sourcemeter_type = pre.get("model") or pre.get("type")
            if not self.Sourcemeter_type and sel.get("resource"):
                self.Sourcemeter_type = (
                    "Prologix" if _looks_like_serial_resource(sel["resource"]) else "VISA"
                )
            return
        use_dummy = getattr(self, "test_mode", None)
        if use_dummy is None:
            use_dummy = JVJTProcedure.test_mode
        selection = getattr(self, "device_selection", None) or JVJTProcedure.device_selection
        self.device_selection = selection
        self.test_mode = use_dummy

        if use_dummy:
            self._connect_dummy()
            return

        if not selection or not selection.get("resource"):
            raise RuntimeError("No instrument selected. Choose Device -> Select Sourcemeter.")

        resource = selection["resource"]
        gpib_address = selection.get("gpib")
        if gpib_address is None:
            gpib_address = DEFAULT_GPIB_ADDRESS

        try:
            if _looks_like_serial_resource(resource):
                visa_resource = _normalize_serial_resource(resource)
                log.info(
                    f"Connecting via Prologix adapter on {visa_resource} (GPIB {gpib_address})."
                )
                self.adapter = PrologixAdapter(
                    visa_resource, gpib_address, gpib_read_timeout=3000
                )
                self.adapter.connection.timeout = 20000  # ms
                # Make absolutely sure the Prologix is configured correctly
                self.adapter.write("++mode 1")  # controller
                self.adapter.write("++auto 0")  # explicit reads
                self.adapter.write("++eoi 1")  # assert EOI with last byte
            else:
                log.info(f"Connecting via VISA resource {resource}.")
                self.adapter = VISAAdapter(resource)
                self.adapter.connection.timeout = 10000  # ms

            self.sourcemeter = Keithley2400(self.adapter)
            model = self._detect_model()
            if model == "2450":
                self._configure_2450_compat()
            self.Sourcemeter_type = model or (
                "Prologix" if _looks_like_serial_resource(resource) else "VISA"
            )
            log.info(f"Connected to instrument type: {self.Sourcemeter_type}")
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to {resource}: {exc}") from exc

    def startup(self):
        log.info("Setting up instrument")
        # prepare data buffers for plotting later
        self.jv_data = []  # will hold (voltage, current) tuples
        self.jt_data = []  # will hold (time, current) tuples
        self.Sourcemeter_type = None
        self._connect_instrument()
        self.sourcemeter.reset()
        if self.max_speed:
            # Pull out all the stops to maximize the speed
            self.nplc_val = 0.01
            self.sourcemeter.write(":DISPlay:DIGits MINimum")
            # digitval = self.sourcemeter.ask(":DISPlay:DIGits?")
            # log.info("Display digits set to: %g" % int(digitval))
            self.sourcemeter.filter_state = "OFF"
            self.sourcemeter.auto_zero = False
            self.sourcemeter.display_enabled = False

        # Configure measurement parameters common to both modes
        self.sourcemeter.measure_current(
            nplc=self.nplc_val, current=0.000105
        )  # Adjust current limit as needed
        sleep(0.1)  # Allow time for settings to apply

        if self.Sourcemeter_type == "2450":
            self.sourcemeter.use_front_terminals()

        self.sourcemeter.stop_buffer()
        self.sourcemeter.disable_buffer()
        self.sourcemeter.write("*OPC?")  # ask the universal Operation Complete bit
        opcreadback = self.sourcemeter.read()  # wait here until the 6430 replies “1”
        log.info(f"OPC read back as:{opcreadback}")
        # Set up some metadata
        self.sm_type_metadata = self.Sourcemeter_type
        self.test_time_metadata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log.info("Instrument setup complete.")

    def execute(self):
        if self.measurement_mode == "JV":
            log.info("Starting JV Measurement")
            # Generate voltage sequence based on sweep mode
            if self.step_size <= 0:
                log.error("Step size must be positive for JV sweep.")
                return  # Stop execution
            if self.sweep_mode == "Standard Sweep":
                num_steps = (
                    int(
                        abs(self.maximum_voltage - self.minimum_voltage)
                        / self.step_size
                    )
                    + 1
                )
                voltages = np.linspace(
                    self.minimum_voltage, self.maximum_voltage, num=num_steps
                )
                log.info(
                    f"Generated standard sweep voltages from {self.minimum_voltage}V to {self.maximum_voltage}V ({num_steps} points)."
                )
            elif self.sweep_mode == "Zero-Centered":
                max_v = self.maximum_voltage
                min_v = self.minimum_voltage
                step = self.step_size

                # Calculate number of steps for each segment, ensuring endpoint inclusion
                # Add small epsilon to avoid floating point issues at boundaries
                epsilon = step * 1e-6
                steps_0_max = (
                    int(round(abs(max_v / step))) if abs(max_v) > epsilon else 0
                )
                steps_max_min = (
                    int(round(abs((max_v - min_v) / step)))
                    if abs(max_v - min_v) > epsilon
                    else 0
                )
                steps_min_0 = (
                    int(round(abs(min_v / step))) if abs(min_v) > epsilon else 0
                )

                # Generate segments using linspace
                v_0_max = np.linspace(0, max_v, steps_0_max + 1)
                v_max_min = np.linspace(max_v, min_v, steps_max_min + 1)[
                    1:
                ]  # Exclude start point (max_v)
                v_min_0 = np.linspace(min_v, 0, steps_min_0 + 1)[
                    1:
                ]  # Exclude start point (min_v)

                voltages = np.concatenate((v_0_max, v_max_min, v_min_0))
                log.info(
                    f"Generated zero-centered sweep voltages: 0V -> {max_v}V -> {min_v}V -> 0V ({len(voltages)} points)."
                )
            else:
                log.error(f"Unknown sweep mode: {self.sweep_mode}")
                return  # Stop execution

            if len(voltages) == 0:
                log.warning("Generated voltage sequence is empty. Check parameters.")
                return  # Stop execution

            # Calculate delay based on sweep speed
            # Delay is the target time between the *start* of setting consecutive voltage points
            delay = (
                max(0, abs(self.step_size / self.sweep_speed))
                if self.sweep_speed != 0
                else 0
            )
            if self.sweep_speed == 0:
                log.warning(
                    "Sweep speed is zero. Measurement will proceed without enforced delay between steps."
                )
            else:
                log.info(f"Target time between voltage steps: {delay:.4f} s")

            self.sourcemeter.enable_source()
            last_step_start_time = time.time()  # Initialize time before the loop
            experiment_start_time = (
                time.time()
            )  # Record the start time of the experiment
            # Loop through each voltage point
            total_steps = len(voltages)
            for count, voltage in enumerate(voltages):
                current_step_start_time = time.time()
                time_since_last_start = current_step_start_time - last_step_start_time

                # Calculate wait time needed before setting voltage to maintain sweep speed
                time_to_wait = max(0, delay - time_since_last_start)
                if time_to_wait > 0:
                    log.debug(
                        f"Waiting for {time_to_wait:.4f} s to maintain sweep speed."
                    )
                    # Interruptible sleep
                    wait_start = time.time()
                    while time.time() - wait_start < time_to_wait:
                        if self.should_stop():
                            break
                        sleep(
                            0.01
                        )  # Short sleep to avoid busy-waiting and allow stop check
                    if self.should_stop():
                        log.warning("User aborted during JV wait interval.")
                        break  # Exit outer loop

                # Check stop condition again after potential wait
                if self.should_stop():
                    log.warning("User aborted the procedure during JV sweep.")
                    break

                # Record the actual start time of processing this step (after waiting)
                last_step_start_time = time.time()

                log.info(
                    f"Step {count + 1}/{total_steps}: Setting voltage to {voltage:.4f} V"
                )
                self.sourcemeter.source_voltage = voltage
                # Measure current
                # sleep(0.1)
                # self.sourcemeter.write(":INIT")
                # self.sourcemeter.write("*WAI")
                # current = float(self.sourcemeter.ask(":FETCH?"))
                current = self.sourcemeter.current
                log.info(f"Measured current: {current:.4e} A")
                elapsed_time = time.time() - experiment_start_time
                data = {
                    "Current JV (A)": current,
                    "Voltage JV (V)": voltage,
                    "Time JV (S)": elapsed_time,
                    "Current JT (A)": np.nan,  # Use NaN for columns not relevant to this mode
                    "Voltage JT (V)": np.nan,
                    "Time JT (S)": np.nan,
                }
                self.emit("results", data)
                self.emit("progress", 100 * (count + 1) / total_steps)
                self.jv_data.append((voltage, current))
            log.info("JV Measurement finished.")

        elif self.measurement_mode == "JT":
            log.info("Starting JT Measurement")
            if self.measurement_interval <= 0:
                log.error("Measurement interval must be positive.")
                return  # Stop execution

            self.sourcemeter.enable_source()
            self.sourcemeter.source_voltage = self.hold_voltage
            log.info(f"Holding voltage at {self.hold_voltage} V")
            # Allow voltage to stabilize
            sleep(0.2)

            start_time = time.time()
            measurement_count = 0

            log.info(f"Measuring at {self.hold_voltage} V.")
            if self.indefinite_measurement:
                log.info(
                    f"Measurement interval: {self.measurement_interval} s. Will continue indefinitely until stopped manually."
                )
            else:
                log.info(
                    f"Measuring for {self.measurement_time} s with interval {self.measurement_interval} s"
                )

            while not self.should_stop():
                loop_iteration_start_time = time.time()

                # Check duration limit if not indefinite
                elapsed_time = loop_iteration_start_time - start_time
                if (
                    not self.indefinite_measurement
                    and elapsed_time >= self.measurement_time
                ):
                    log.info("Measurement duration reached.")
                    break  # Exit loop if total time exceeded

                # Measure current
                current = self.sourcemeter.current
                measurement_end_time = time.time()
                actual_measurement_time_point = (
                    measurement_end_time - start_time
                )  # Time relative to start

                data = {
                    "Current JV (A)": np.nan,
                    "Voltage JV (V)": np.nan,
                    "Time JV (S)": np.nan,
                    "Current JT (A)": current,
                    "Voltage JT (V)": self.hold_voltage,
                    "Time JT (S)": actual_measurement_time_point,
                }
                self.emit("results", data)

                # Progress reporting (only if not indefinite)
                if not self.indefinite_measurement:
                    progress = min(
                        100,
                        int(
                            100
                            * (actual_measurement_time_point / self.measurement_time)
                        ),
                    )
                    self.emit("progress", progress)
                self.jt_data.append((actual_measurement_time_point, current))
                measurement_count += 1

                # Calculate the time until the next measurement should ideally start
                next_ideal_start_time = (
                    start_time + measurement_count * self.measurement_interval
                )
                # Calculate how long to wait from now
                wait_time = next_ideal_start_time - time.time()

                if wait_time > 0:
                    log.debug(f"Waiting for {wait_time:.4f} s until next measurement.")
                    # Interruptible sleep
                    wait_start = time.time()
                    while time.time() - wait_start < wait_time:
                        if self.should_stop():
                            break
                        sleep(0.01)  # Short sleep to avoid busy-waiting
                    if self.should_stop():
                        log.warning("User aborted during JT wait interval.")
                        break  # Break outer measurement loop
                else:
                    # Log if we are lagging significantly behind the desired interval
                    if wait_time < -0.1:  # Log if more than 100ms behind schedule
                        log.warning(
                            f"Measurement/processing took too long. Cannot maintain exact interval of {self.measurement_interval} s. Lagging by {-wait_time:.4f} s."
                        )
                    # No sleep needed, proceed to next measurement immediately

            if self.should_stop() and not (
                not self.indefinite_measurement
                and elapsed_time >= self.measurement_time
            ):
                log.warning("User aborted the procedure during JT measurement.")
            log.info("JT Measurement finished.")

    def shutdown(self):
        if hasattr(self, "sourcemeter"):
            self.sourcemeter.disable_source()
            log.info("Source disabled.")
            # self.sourcemeter.shutdown()
            log.info("Instrument shutdown procedure called.")
        if hasattr(self, "adapter"):
            self.adapter.close()
            log.info("Adapter closed.")
        log.info("Finished measuring.")

        # Now build and save matplotlib plots headlessly
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath("Output")
        os.makedirs(out_dir, exist_ok=True)

        try:
            if self.jv_data:
                volts, amps = zip(*self.jv_data)
                fig, ax = plt.subplots()
                ax.plot(volts, amps, marker="o", linestyle="-")
                ax.set_xlabel("Voltage (V)")
                ax.set_ylabel("Current (A)")
                ax.set_title(f"{self.identifier} JV Sweep")
                fname = f"{self.identifier}_JV_{timestamp}.png"
                path = os.path.join(out_dir, fname)
                fig.savefig(path)
                plt.close(fig)
                log.info(f"Saved JV plot → {path}")

            if self.jt_data:
                times, amps = zip(*self.jt_data)
                fig, ax = plt.subplots()
                ax.plot(times, amps, marker="o", linestyle="-")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Current (A)")
                ax.set_title(f"{self.identifier} JT Measurement")
                fname = f"{self.identifier}_JT_{timestamp}.png"
                path = os.path.join(out_dir, fname)
                fig.savefig(path)
                plt.close(fig)
                log.info(f"Saved JT plot → {path}")

        except Exception as e:
            log.error(f"Failed to save plot: {e}")


class MainWindow(ManagedDockWindow):
    def __init__(self):
        super().__init__(
            procedure_class=JVJTProcedure,
            inputs=[
                "measurement_mode",
                "indefinite_measurement",
                "max_speed",
                "nplc_val",
                "minimum_voltage",
                "maximum_voltage",
                "step_size",
                "sweep_speed",
                "sweep_mode",
                "hold_voltage",
                "measurement_time",
                "measurement_interval",
                "identifier",
            ],
            displays=[
                "measurement_mode",
                "nplc_val",
                "minimum_voltage",
                "maximum_voltage",
                "step_size",
                "sweep_speed",
                "sweep_mode",
                "hold_voltage",
                "measurement_time",
                "measurement_interval",
            ],
            x_axis=["Voltage JV (V)", "Time JT (S)"],
            y_axis=["Current JV (A)", "Current JT (A)"],
            linewidth=3,
        )
        self.setWindowTitle("Keithley Control")
        icon_path = resource_path("res/icons/Appicon.png")  # New line
        self.setWindowIcon(QIcon(icon_path))  # New line
        self.directory = r"Output"

        self.filename = r"{Identifier}_{Measurement mode}_{date}"
        self.allow_dummy = ALLOW_DUMMY
        self.device_selection = None
        self.device_idn = None
        self.test_mode = False
        self._dummy_action = None
        JVJTProcedure.device_selection = None
        JVJTProcedure.preconnected = None
        JVJTProcedure.test_mode = False
        self._setup_menu()
        self._update_device_status()

    def _setup_menu(self):
        # Get the menu bar provided by QMainWindow
        menu_bar = self.menuBar()

        # Create File menu
        file_menu = menu_bar.addMenu("&File")

        # Add Exit action
        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.triggered.connect(
            self.close
        )  # Connect to the window's close method
        # Or connect directly to app quit: exit_action.triggered.connect(QtWidgets.QApplication.instance().quit)
        file_menu.addAction(exit_action)

        device_menu = menu_bar.addMenu("&Device")
        select_device = QtWidgets.QAction("Select Sourcemeter...", self)
        select_device.triggered.connect(self._prompt_device_selection)
        device_menu.addAction(select_device)

        if self.allow_dummy:
            self._dummy_action = QtWidgets.QAction("Use Dummy Sourcemeter", self)
            self._dummy_action.setCheckable(True)
            self._dummy_action.setChecked(self.test_mode)
            self._dummy_action.toggled.connect(self._toggle_dummy_mode)
            device_menu.addAction(self._dummy_action)
        self._update_device_status()

    def _update_device_status(self):
        if self.test_mode:
            status = "Dummy sourcemeter (test mode)"
        elif not self.device_selection:
            status = "No sourcemeter selected. Use Device -> Select Sourcemeter."
        else:
            status = f"Sourcemeter: {self.device_selection.get('resource')}"
            gpib_val = self.device_selection.get("gpib")
            if gpib_val is not None:
                status += f" (GPIB {gpib_val})"
            if self.device_idn:
                status += f" [{self.device_idn}]"
        self.statusBar().showMessage(status)

    def _switch_to_dummy_mode(self, *, notify=False, reason=None):
        if not self.allow_dummy:
            return
        self.device_selection = None
        self.device_idn = None
        self.test_mode = True
        JVJTProcedure.device_selection = None
        JVJTProcedure.preconnected = None
        JVJTProcedure.test_mode = True
        if self._dummy_action and not self._dummy_action.isChecked():
            self._dummy_action.setChecked(True)
        if notify and reason:
            QtWidgets.QMessageBox.information(self, "Dummy mode", reason)
        self._update_device_status()

    def _toggle_dummy_mode(self, checked):
        if not self.allow_dummy:
            return
        if checked:
            self._switch_to_dummy_mode()
            return
        # User unchecked dummy mode – prompt for a real device
        self.test_mode = False
        JVJTProcedure.test_mode = False
        self._prompt_device_selection()

    def _prompt_device_selection(self):
        dialog = DevicePicker(
            self,
            title="Select sourcemeter",
            show_gpib=True,
            default_gpib=DEFAULT_GPIB_ADDRESS,
        )
        if self.device_selection:
            dialog.manual.setText(self.device_selection.get("resource", ""))
            if self.device_selection.get("gpib") is not None:
                dialog.gpib_spin.setValue(self.device_selection["gpib"])
        result = _dialog_exec(dialog)
        if result == QtWidgets.QDialog.Accepted:
            resource, gpib = dialog.get()
            if resource:
                self._connect_selected_device(resource, gpib)
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No device selected",
                    "Select a VISA resource or cancel to keep current settings.",
                )
        self._update_device_status()

    def _connect_selected_device(self, resource, gpib):
        """Connect immediately when the picker is used and announce the ID."""
        self.device_selection = {"resource": resource, "gpib": gpib}
        self.device_idn = None
        self.test_mode = False
        JVJTProcedure.device_selection = self.device_selection
        JVJTProcedure.test_mode = False
        prev = JVJTProcedure.preconnected
        if prev and prev.get("adapter"):
            try:
                prev["adapter"].close()
            except Exception:
                pass
        JVJTProcedure.preconnected = None
        if self._dummy_action and self._dummy_action.isChecked():
            self._dummy_action.setChecked(False)

        adapter = None
        sourcemeter = None
        ident = "Unknown"
        model = None
        try:
            if resource.upper().startswith(("USB", "GPIB", "TCPIP")):
                adapter = VISAAdapter(resource)
                adapter.connection.timeout = 10000
            else:
                visa_resource = _normalize_serial_resource(resource)
                adapter = PrologixAdapter(
                    visa_resource, gpib or DEFAULT_GPIB_ADDRESS, gpib_read_timeout=3000
                )
                adapter.connection.timeout = 20000
                adapter.write("++mode 1")
                adapter.write("++auto 0")
                adapter.write("++eoi 1")

            sourcemeter = Keithley2400(adapter)
            # Basic setup/zero similar to the provided snippet
            sourcemeter.reset()
            try:
                ident = sourcemeter.ask("*IDN?") or ident
            except Exception:
                pass
            if "2450" in ident:
                try:
                    sourcemeter.write("SYST:LANG SCPI")
                    sourcemeter.write("SYST:CONS 2400")
                except Exception:
                    pass
            sourcemeter.apply_voltage()
            sourcemeter.source_voltage_range = 100
            sourcemeter.source_voltage = 0
            sourcemeter.enable_source()
            sourcemeter.measure_current(nplc=1, current=0.105, auto_range=False)
            sourcemeter.current_range = 0.001
            sourcemeter.disable_source()
            model = None
            for candidate in ("2450", "2400", "6430"):
                if candidate in ident:
                    model = candidate
                    break

            self.device_idn = ident.strip()
            JVJTProcedure.preconnected = {
                "sourcemeter": sourcemeter,
                "adapter": adapter,
                "model": model,
                "selection": self.device_selection,
            }
            log.info("Connected sourcemeter: %s", self.device_idn)
            QtWidgets.QMessageBox.information(
                self,
                "Sourcemeter",
                f"Connected: {ident}\nOutput zeroed and disabled.",
            )
        except Exception as exc:
            JVJTProcedure.preconnected = None
            self.device_selection = None
            JVJTProcedure.device_selection = None
            self.device_idn = None
            if adapter:
                try:
                    adapter.close()
                except Exception:
                    pass
            QtWidgets.QMessageBox.critical(
                self, "Sourcemeter", f"Failed to connect: {exc}"
            )
        self._update_device_status()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
