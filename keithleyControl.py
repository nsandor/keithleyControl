import logging
import numpy as np
import sys
import os
import time
from time import sleep

# -- Force a non‐GUI backend before importing pyplot --
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # For creating and saving your own plots

from pymeasure.log import log
from pymeasure.adapters import VISAAdapter, PrologixAdapter
from drivers.dummy_keithley import DummyKeithley2400
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
    Metadata,
)
from PyQt5.QtGui import QIcon

# Testing mode, uses dummy driver
test = True

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

class JVJTProcedure(Procedure):
    measurement_mode = ListParameter(
        "Measurement mode", ["JV", "JT"], default="JV"
    )
    identifier = Parameter("Identifier", default="Device")
    max_speed = BooleanParameter("Maximize Measurement Speed", default=False)
    nplc_val = FloatParameter(
        "NPLC value", default=10, group_by="max_speed", group_condition=False
    )

    # JV parameters
    minimum_voltage = FloatParameter(
        "Minimum Voltage", units="V", default=-10,
        group_by="measurement_mode", group_condition="JV"
    )
    maximum_voltage = FloatParameter(
        "Maximum Voltage", units="V", default=10,
        group_by="measurement_mode", group_condition="JV"
    )
    step_size = FloatParameter(
        "Step Size", units="V", default=1,
        group_by="measurement_mode", group_condition="JV"
    )
    sweep_speed = FloatParameter(
        "Sweep Speed", units="V/s", default=1,
        group_by="measurement_mode", group_condition="JV"
    )
    sweep_mode = ListParameter(
        "Sweep Type", ["Standard Sweep", "Zero-Centered"], default="Standard Sweep",
        group_by="measurement_mode", group_condition="JV"
    )

    # JT parameters
    hold_voltage = FloatParameter(
        "Hold Voltage", units="V", default=1,
        group_by="measurement_mode", group_condition="JT"
    )
    indefinite_measurement = BooleanParameter(
        "Indefinite Measurement", default=False,
        group_by="measurement_mode", group_condition="JT"
    )
    measurement_time = IntegerParameter(
        "Measurement Duration", units="S", default=30,
        group_by={"measurement_mode": "JT", "indefinite_measurement": False}
    )
    measurement_interval = FloatParameter(
        "Measurement Interval", units="S", default=1,
        group_by="measurement_mode", group_condition="JT"
    )

    DATA_COLUMNS = [
        "Current JV (A)", "Voltage JV (V)", "Time JV (S)",
        "Current JT (A)", "Voltage JT (V)", "Time JT (S)",
    ]

    sm_type_metadata = Metadata("Sourcemeter Type", default="None")
    test_time_metadata = Metadata("Test Time", default="None")

    def startup(self):
        log.info("Setting up instrument")
        self.Sourcemeter_type = None

        # prepare data buffers for plotting later
        self.jv_data = []   # will hold (voltage, current) tuples
        self.jt_data = []   # will hold (time, current) tuples

        if test:
            self.sourcemeter = DummyKeithley2400()
            self.Sourcemeter_type = "Dummy"
            log.info("Connected to dummy instrument.")
        else:
            try:
                self.adapter = PrologixAdapter("ASRL4::INSTR", 7, gpib_read_timeout=3000)
                self.sourcemeter = Keithley2400(self.adapter)
                self.Sourcemeter_type = "6430"
                log.info("Connected via Prologix GPIB.")
            except Exception:
                log.info("Prologix not found, trying VISA...")
            if self.Sourcemeter_type is None:
                try:
                    self.adapter = VISAAdapter("USB0::0x05E6::0x2450::04491080::INSTR")
                    self.sourcemeter = Keithley2400(self.adapter)
                    self.Sourcemeter_type = "2450"
                    log.info("Connected via VISA adapter.")
                except Exception:
                    raise RuntimeError("No sourcemeter found. Please check connection.")

        self.sourcemeter.reset()
        if self.max_speed:
            self.nplc_val = 0.01
            self.sourcemeter.write(":DISPlay:DIGits MINimum")
            self.sourcemeter.filter_state = "OFF"
            self.sourcemeter.auto_zero = False
            self.sourcemeter.display_enabled = False

        self.sourcemeter.measure_current(nplc=self.nplc_val)
        sleep(0.1)
        if self.Sourcemeter_type == "2450":
            self.sourcemeter.use_rear_terminals()

        self.sourcemeter.stop_buffer()
        self.sourcemeter.disable_buffer()

        self.sm_type_metadata = self.Sourcemeter_type
        self.test_time_metadata = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log.info("Startup complete.")

    def chime(self):
        if hasattr(self.sourcemeter, "triad"):
            try:
                for freq in (261.63, 440.00, 349.23):  # C, A, F
                    self.sourcemeter.triad(freq, 0.5)
                    time.sleep(0.5)
            except Exception as e:
                log.warning(f"Chime failed: {e}")

    def execute(self):
        if self.measurement_mode == "JV":
            log.info("Running JV sweep")
            # build voltage array
            if self.step_size <= 0:
                log.error("Step size must be > 0.")
                return
            if self.sweep_mode == "Standard Sweep":
                num = int(abs(self.maximum_voltage - self.minimum_voltage) / self.step_size) + 1
                voltages = np.linspace(self.minimum_voltage, self.maximum_voltage, num=num)
            else:  # zero-centered
                eps = self.step_size * 1e-6
                max_v, min_v, step = self.maximum_voltage, self.minimum_voltage, self.step_size
                v1 = np.linspace(0, max_v, int(round(abs(max_v/step)))+1) if abs(max_v)>eps else []
                v2 = np.linspace(max_v, min_v, int(round(abs((max_v-min_v)/step)))+1)[1:] if abs(max_v-min_v)>eps else []
                v3 = np.linspace(min_v, 0, int(round(abs(min_v/step)))+1)[1:] if abs(min_v)>eps else []
                voltages = np.concatenate((v1, v2, v3))

            delay = abs(self.step_size/self.sweep_speed) if self.sweep_speed>0 else 0
            self.sourcemeter.enable_source()
            t0 = time.time()
            last = time.time()

            for idx, v in enumerate(voltages):
                dt = time.time() - last
                if dt < delay:
                    sleep(delay - dt)
                last = time.time()

                self.sourcemeter.source_voltage = v
                i = self.sourcemeter.current
                elapsed = time.time() - t0

                self.emit("results", {
                    "Current JV (A)": i, "Voltage JV (V)": v, "Time JV (S)": elapsed,
                    "Current JT (A)": np.nan, "Voltage JT (V)": np.nan, "Time JT (S)": np.nan
                })
                self.emit("progress", 100*(idx+1)/len(voltages))

                # collect for offline plotting
                self.jv_data.append((v, i))

            self.chime()
            log.info("JV done.")

        else:  # JT mode
            log.info("Running JT measurement")
            if self.measurement_interval <= 0:
                log.error("Interval must be > 0.")
                return

            self.sourcemeter.enable_source()
            self.sourcemeter.source_voltage = self.hold_voltage
            sleep(0.2)
            t0 = time.time()
            count = 0

            while True:
                elapsed = time.time() - t0
                if not self.indefinite_measurement and elapsed >= self.measurement_time:
                    break

                i = self.sourcemeter.current
                self.emit("results", {
                    "Current JV (A)": np.nan, "Voltage JV (V)": np.nan, "Time JV (S)": np.nan,
                    "Current JT (A)": i, "Voltage JT (V)": self.hold_voltage, "Time JT (S)": elapsed
                })
                if not self.indefinite_measurement:
                    self.emit("progress", int(100*elapsed/self.measurement_time))

                # collect for offline plotting
                self.jt_data.append((elapsed, i))
                count += 1

                next_t = t0 + count*self.measurement_interval
                dt = next_t - time.time()
                if dt > 0:
                    sleep(dt)

            self.chime()
            log.info("JT done.")

    def shutdown(self):
        # turn everything off
        if hasattr(self, "sourcemeter"):
            self.sourcemeter.disable_source()
            self.sourcemeter.shutdown()
        if hasattr(self, "adapter"):
            self.adapter.close()

        # Now build and save matplotlib plots headlessly
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath("Output")
        os.makedirs(out_dir, exist_ok=True)

        try:
            if self.jv_data:
                volts, amps = zip(*self.jv_data)
                fig, ax = plt.subplots()
                ax.plot(volts, amps, marker='o', linestyle='-')
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
                ax.plot(times, amps, marker='o', linestyle='-')
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

        log.info("Shutdown complete.")

class MainWindow(ManagedDockWindow):
    def __init__(self):
        super().__init__(
            procedure_class=JVJTProcedure,
            inputs=[
                "measurement_mode", "indefinite_measurement", "max_speed", "nplc_val",
                "minimum_voltage", "maximum_voltage", "step_size", "sweep_speed",
                "sweep_mode", "hold_voltage", "measurement_time",
                "measurement_interval", "identifier"
            ],
            displays=[
                "measurement_mode", "nplc_val", "minimum_voltage", "maximum_voltage",
                "step_size", "sweep_speed", "sweep_mode", "hold_voltage",
                "measurement_time", "measurement_interval"
            ],
            x_axis=["Voltage JV (V)", "Time JT (S)"],
            y_axis=["Current JV (A)", "Current JT (A)"],
            linewidth=3,
        )
        self.setWindowTitle("Keithley Control")
        icon_path = resource_path("res/icons/Appicon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.directory = r"Output"
        self.filename = r"{Identifier}_{Measurement mode}_{date}"
        self._setup_menu()

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
