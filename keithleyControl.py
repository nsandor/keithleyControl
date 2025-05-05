import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
import numpy as np
import sys
import time  # Import the time module
from time import sleep
from pymeasure.log import log
from pymeasure.adapters import VISAAdapter, PrologixAdapter
from drivers.dummy_keithley import DummyKeithley2400

# Both the 6430 and 2450 use essentially the same commands, so the 2400 driver works fine
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows.managed_dock_window import ManagedDockWindow
from pymeasure.experiment import Procedure
from pymeasure.experiment import (
    Parameter,
    IntegerParameter,
    FloatParameter,
    ListParameter,
    BooleanParameter,
    Metadata
)

# Testing mode, uses dummy driver
test = True


class JVJTProcedure(Procedure):
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

    def startup(self):
        log.info("Setting up instrument")
        self.Sourcemeter_type = None
        if test:
            self.sourcemeter = DummyKeithley2400()  # Dummy instrument
            self.Connected = True
            self.Sourcemeter_type = "Dummy"
            log.info("Connected to instrument.")
        else:
            try:
                self.adapter = PrologixAdapter(
                    "ASRL4::INSTR", 7, gpib_read_timeout=3000
                )
                self.sourcemeter = Keithley2400(self.adapter)
                self.Sourcemeter_type = "6430"
                log.info("Connected to Prologix adapter.")
            except Exception:
                log.info(
                    "No Prologix adapter found (no 6400 here), trying VISA adapter, maybe the 2450 is connected?."
                )
            if self.Sourcemeter_type is None:
                try:
                    # Attempt to use VISA adapter if available
                    self.adapter = VISAAdapter("USB0::0x05E6::0x2450::04491080::INSTR")
                    self.sourcemeter = Keithley2400(self.adapter)
                    self.Sourcemeter_type = "2450"
                    log.info("Connected to VISA adapter.")
                except Exception:
                    log.info("No sourcemeter found, is one plugged in?")
                    # If no instrument is found, raise an error
                    raise RuntimeError(
                        "No instrument found. Please check the connection."
                    )
        sourcemeter_type = Metadata("sourcemeter_type",default=self.Sourcemeter_type)
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
            nplc=self.nplc_val
        )  # Adjust current limit as needed
        sleep(0.1)  # Allow time for settings to apply

        if self.Sourcemeter_type == "2450":
            self.sourcemeter.use_rear_terminals()

        self.sourcemeter.stop_buffer()
        self.sourcemeter.disable_buffer()
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
                    f"Step {count+1}/{total_steps}: Setting voltage to {voltage:.4f} V"
                )
                self.sourcemeter.source_voltage = voltage
                # Measure current
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
                measurement_start_time = time.time()
                current = self.sourcemeter.current
                measurement_end_time = time.time()
                actual_measurement_time_point = (
                    measurement_end_time - start_time
                )  # Time relative to start

                log.info(
                    f"Time: {actual_measurement_time_point:.2f}s, Measured Current: {current:.4e} A"
                )

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
            self.sourcemeter.shutdown()
            log.info("Instrument shutdown procedure called.")
        if hasattr(self, "adapter"):
            self.adapter.close()
            log.info("Adapter closed.")
        log.info("Finished measuring.")


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
        self.setWindowTitle("GUI Example")
        self.directory = r"Output"

        self.filename = r"{Identifier}_{Measurement mode}_{date}_{time}"
        self._setup_menu()

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

        # You can add other menus (Edit, View, Help) and actions here
        # help_menu = menu_bar.addMenu("&Help")
        # about_action = QtWidgets.QAction("&About", self)
        # help_menu.addAction(about_action)
        # # Connect about_action.triggered to a function that shows an about dialog


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
