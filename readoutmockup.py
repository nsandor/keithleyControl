#!/usr/bin/env python3
# photodiode_array_control.py  –  2025‑07‑15
#
# • Two Keithleys (real or Dummy) – bias + readout (readout locked to 0 V)
# • USB 100:1 switch (Nano Every firmware v1.0)
# • 10×10 pixel grid (Select / Measure)
# • Sequence dock: queue JV/JT over any pixel range
#
# ----------------------------------------------------------------------

import logging
import sys
import time
from pathlib import Path
from typing import Optional
from time import sleep

import numpy as np
import serial
from pymeasure.adapters import PrologixAdapter, VISAAdapter
from pymeasure.display.Qt import QtWidgets, QtCore, QtGui
from pymeasure.display.windows.managed_dock_window import ManagedDockWindow
from pymeasure.experiment import (BooleanParameter, FloatParameter,
                                  IntegerParameter, ListParameter, Metadata,
                                  Parameter, Procedure)
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.log import log

# Dummy Keithley driver supplied by the user’s project
from drivers.dummy_keithley import DummyKeithley2400

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

def safe_id(instr):
    try:
        return instr.id          # SCPI instruments
    except (AttributeError, NotImplementedError):
        return instr.__class__.__name__

# ----------------------------------------------------------------------
# Qt helpers  (binding‑agnostic)
# ----------------------------------------------------------------------
Signal = QtCore.Signal if hasattr(QtCore, "Signal") else QtCore.pyqtSignal
if hasattr(QtWidgets, "QTableWidget"):        # Qt ≥ 5
    QTableWidget = QtWidgets.QTableWidget
    QTableWidgetItem = QtWidgets.QTableWidgetItem
else:                                         # Qt4 fall‑back
    QTableWidget = QtGui.QTableWidget
    QTableWidgetItem = QtGui.QTableWidgetItem

# ----------------------------------------------------------------------
# Switch‑board serial wrapper
# ----------------------------------------------------------------------
class SwitchBoard:
    """USB switch board – firmware v1.0 (Nano Every)."""

    def __init__(self, port: str, baud: int = 9600, timeout: float = 1):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.verbose = True

    def _tx(self, cmd: str) -> str:
        self.ser.write((cmd + "\n").encode())
        return self.ser.readline().decode(errors="replace").strip()

    # ---- commands ------------------------------------------------------
    def route(self, index: int):
        if not 1 <= index <= 100:
            raise ValueError("Pixel index must be 1‑100")
        self._tx(str(index))

    def set_verbose(self, on: bool):
        self.verbose = on
        self._tx(f"VERBOSE {'ON' if on else 'OFF'}")

    def led(self, on: bool):          self._tx(f"LED {'ON' if on else 'OFF'}")
    def tia(self, on: bool):          self._tx(f"TIA {'ON' if on else 'OFF'}")
    def amp(self, on: bool):          self._tx(f"AMP {'ON' if on else 'OFF'}")
    def route_out(self, on: bool):    self._tx(f"ROUTE {'ON' if on else 'OFF'}")
    def adc_read(self) -> str:        return self._tx("ADC")
    def spi_write(self, b1: int, b2: int): self._tx(f"SPI {b1:02X} {b2:02X}")
    def close(self): self.ser.close()

# ----------------------------------------------------------------------
# Dummy switch for offline testing
# ----------------------------------------------------------------------
class SwitchBoardDummy:
    def route(self, *_):      pass
    def set_verbose(self, _): pass
    def led(self, _):         pass
    def tia(self, _):         pass
    def amp(self, _):         pass
    def route_out(self, _):   pass
    def adc_read(self):       return "0  0.000 V"
    def spi_write(self, *_):  pass
    def close(self):          pass

# ----------------------------------------------------------------------
# Readout‑safe sourcemeter
# ----------------------------------------------------------------------
class ReadoutSafe2400(Keithley2400):
    """Keithley that is hard‑locked to 0 V sourcing."""

    def __init__(self, adapter):
        super().__init__(adapter)
        self.disable_source()
        self.write(":SOUR:FUNC VOLT")
        super(ReadoutSafe2400, self.__class__).source_voltage.fset(self, 0)

    @property
    def source_voltage(self):
        return float(self.ask(":SOUR:VOLT?"))

    @source_voltage.setter
    def source_voltage(self, val):
        if val != 0:
            raise RuntimeError("Readout SM must remain at 0 V")
        super(ReadoutSafe2400, self.__class__).source_voltage.fset(self, 0)

    def enable_source(self):
        if self.source_voltage != 0:
            raise RuntimeError("Readout SM enable refused – voltage ≠ 0 V")
        super().enable_source()

# ----------------------------------------------------------------------
# Measurement Procedure
# ----------------------------------------------------------------------
class JVJTProcedure(Procedure):
    # parameters
    measurement_mode = ListParameter("Measurement mode", ["JV", "JT"], default="JV")
    identifier       = Parameter("Identifier", default="Device")
    pixel_index      = IntegerParameter("Pixel (1‑100)", default=1)

    max_speed = BooleanParameter("Maximize Measurement Speed", default=False)
    nplc_val  = FloatParameter("NPLC value", default=10,
                               group_by="max_speed", group_condition=False)

    # JV
    minimum_voltage = FloatParameter("Minimum Voltage", units="V",
                                     default=-1, group_by="measurement_mode", group_condition="JV")
    maximum_voltage = FloatParameter("Maximum Voltage", units="V",
                                     default=1, group_by="measurement_mode", group_condition="JV")
    step_size = FloatParameter("Step Size", units="V",
                               default=0.1, group_by="measurement_mode", group_condition="JV")
    sweep_speed = FloatParameter("Sweep Speed", units="V/s",
                                 default=0.2, group_by="measurement_mode", group_condition="JV")
    sweep_mode = ListParameter("Sweep Type", ["Standard Sweep", "Zero-Centered"],
                               default="Standard Sweep",
                               group_by="measurement_mode", group_condition="JV")

    # JT
    hold_voltage = FloatParameter("Hold Voltage", units="V",
                                  default=0.5, group_by="measurement_mode", group_condition="JT")
    indefinite_measurement = BooleanParameter("Indefinite Measurement", default=False,
                                              group_by="measurement_mode", group_condition="JT")
    measurement_time = IntegerParameter("Measurement Duration", units="S", default=30,
                                        group_by={"measurement_mode": "JT",
                                                  "indefinite_measurement": False})
    measurement_interval = FloatParameter("Measurement Interval", units="S", default=1,
                                          group_by="measurement_mode", group_condition="JT")

    DATA_COLUMNS = ["Current (A)", "Voltage (V)", "Time (s)"]
    sm_type_metadata  = Metadata("Sourcemeter Type", default="None")
    test_time_metadata = Metadata("Test Time", default="None")

    # handles (set by MainWindow.make_procedure)
    bias_sm  = None
    read_sm  = None
    switch   = None

    # ------------------------------------------------------------------
    def startup(self):
        if not (self.bias_sm and self.read_sm and self.switch):
            raise RuntimeError("Hardware handles missing")

        for sm in (self.bias_sm, self.read_sm):
            sm.measure_current(nplc=self.nplc_val)
        if self.max_speed:
            self.nplc_val = 0.01
            self.bias_sm.filter_state = "OFF"
            self.bias_sm.auto_zero = False
            self.bias_sm.display_enabled = False
        self.sm_type_metadata = f"{safe_id(self.bias_sm)} (bias) + {safe_id(self.read_sm)} (read)"
        self.test_time_metadata = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Procedure startup complete")

    # ------------------------------------------------------------------
    def execute(self):
        self.switch.route(self.pixel_index)
        sleep(0.05)
        if self.measurement_mode == "JV":
            self._run_jv()
        else:
            self._run_jt()

    def _run_jv(self):
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")
        voltages = np.linspace(self.minimum_voltage, self.maximum_voltage,
                               int(abs(self.maximum_voltage - self.minimum_voltage) /
                                   self.step_size) + 1)
        delay = abs(self.step_size / self.sweep_speed) if self.sweep_speed else 0
        start = time.time()
        self.bias_sm.enable_source()
        for i, v in enumerate(voltages):
            if self.should_stop(): break
            self.bias_sm.source_voltage = v
            self.bias_sm.write(":INIT;*WAI")
            self.read_sm.enable_source()
            cur = float(self.read_sm.ask(":READ?"))
            self.read_sm.disable_source()
            self.emit("results", {"Current (A)": cur,
                                  "Voltage (V)": v, "Time (s)": time.time() - start})
            self.emit("progress", 100*(i+1)/len(voltages))
            sleep(delay)
        self.bias_sm.disable_source()

    def _run_jt(self):
        self.bias_sm.enable_source()
        self.bias_sm.source_voltage = self.hold_voltage
        start = time.time(); count = 0
        while not self.should_stop():
            t_rel = time.time() - start
            if not self.indefinite_measurement and t_rel >= self.measurement_time:
                break
            self.read_sm.enable_source()
            cur = float(self.read_sm.current)
            self.read_sm.disable_source()
            self.emit("results", {"Current (A)": cur,
                                  "Voltage (V)": self.hold_voltage, "Time (s)": t_rel})
            if not self.indefinite_measurement:
                self.emit("progress", 100*t_rel/self.measurement_time)
            count += 1
            sleep(max(0, start + count*self.measurement_interval - time.time()))
        self.bias_sm.disable_source()

    def shutdown(self): logger.info("Procedure done")

# ----------------------------------------------------------------------
# Pixel grid widget
# ----------------------------------------------------------------------
class PixelGridWidget(QtWidgets.QWidget):
    pixelClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        table = QTableWidget(10, 10)
        table.verticalHeader().hide()
        table.horizontalHeader().hide()
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        for r in range(10):
            for c in range(10):
                idx = r * 10 + c + 1
                item = QTableWidgetItem(str(idx))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(r, c, item)

        table.cellClicked.connect(
            lambda r, c: self.pixelClicked.emit(r * 10 + c + 1)
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)

# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class MainWindow(ManagedDockWindow):
    def __init__(self):
        super().__init__(
            procedure_class=JVJTProcedure,
            inputs=["measurement_mode", "indefinite_measurement", "max_speed", "nplc_val",
                    "minimum_voltage", "maximum_voltage", "step_size", "sweep_speed",
                    "sweep_mode", "hold_voltage", "measurement_time",
                    "measurement_interval", "identifier", "pixel_index"],
            displays=["measurement_mode", "pixel_index", "nplc_val", "minimum_voltage",
                      "maximum_voltage", "step_size", "sweep_speed", "sweep_mode",
                      "hold_voltage", "measurement_time", "measurement_interval"],
            x_axis=["Voltage (V)", "Time (s)"],
            y_axis=["Current (A)"],
            linewidth=2,
        )

        self.setWindowTitle("Photodiode Array – Keithley Control")
        self.directory = "Output"
        self.setWindowIcon(QtGui.QIcon(resource_path("res/icons/Appicon.png")))

        # Handles
        self.bias_sm: Optional[Keithley2400]    = None
        self.read_sm: Optional[ReadoutSafe2400] = None
        self.switch : Optional[SwitchBoard]     = None

        # UI
        self._build_pixel_dock()
        self._build_switch_control_dock()
        self._build_sequence_dock()
        self._build_hw_menu()

    # ------------------------------------------------------------------ injection
    def make_procedure(self):
        """Create a procedure and inject hardware handles."""
        proc: JVJTProcedure = super().make_procedure()
        proc.bias_sm = self.bias_sm or DummyKeithley2400()
        proc.read_sm = self.read_sm or DummyKeithley2400()
        proc.switch  = self.switch  or SwitchBoardDummy()
        return proc

    # ------------------------------------------------------------------ UI builders
    def _build_pixel_dock(self):
        dock = QtWidgets.QDockWidget("Pixel Array")
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        host = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(host)

        self.rb_select  = QtWidgets.QRadioButton("Select")
        self.rb_measure = QtWidgets.QRadioButton("Measure")
        self.rb_select.setChecked(True)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.rb_select); h.addWidget(self.rb_measure)
        vbox.addLayout(h)

        grid = PixelGridWidget(); grid.pixelClicked.connect(self._on_pixel_clicked)
        vbox.addWidget(grid)
        dock.setWidget(host)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_switch_control_dock(self):
        dock = QtWidgets.QDockWidget("Switch Controls")
        host = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(host)

        self.chk_verbose = QtWidgets.QCheckBox("Verbose"); self.chk_verbose.setChecked(True)
        self.chk_verbose.stateChanged.connect(lambda s:
            self.switch and self.switch.set_verbose(bool(s)))
        form.addRow(self.chk_verbose)

        self.chk_led = QtWidgets.QCheckBox("Panel LED")
        self.chk_led.stateChanged.connect(lambda s:
            self.switch and self.switch.led(bool(s)))
        form.addRow(self.chk_led)

        for label, cmd in (("TIA", "tia"), ("AMP", "amp"), ("ROUTE OUT", "route_out")):
            chk = QtWidgets.QCheckBox(label)
            chk.stateChanged.connect(lambda s, c=cmd:
                getattr(self.switch, c)(bool(s)) if self.switch else None)
            form.addRow(chk)

        spi_line = QtWidgets.QLineEdit("3F A0")
        spi_btn = QtWidgets.QPushButton("Send SPI")
        spi_btn.clicked.connect(lambda: self._send_spi(spi_line.text()))
        form.addRow(spi_btn, spi_line)

        adc_btn = QtWidgets.QPushButton("Read ADC (A7)")
        adc_lbl = QtWidgets.QLabel("–")
        adc_btn.clicked.connect(lambda:
            adc_lbl.setText(self.switch.adc_read() if self.switch else "N/A"))
        form.addRow(adc_btn, adc_lbl)

        dock.setWidget(host)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_sequence_dock(self):
        dock = QtWidgets.QDockWidget("Measurement Sequence")
        host = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(host)

        self.seq_start = QtWidgets.QSpinBox(); self.seq_start.setRange(1, 100); self.seq_start.setValue(1)
        self.seq_end   = QtWidgets.QSpinBox(); self.seq_end.setRange(1, 100); self.seq_end.setValue(100)
        form.addRow("Start Pixel", self.seq_start)
        form.addRow("End Pixel",   self.seq_end)

        btn_jv = QtWidgets.QPushButton("Queue JV Sequence")
        btn_jt = QtWidgets.QPushButton("Queue JT Sequence")
        btn_jv.clicked.connect(lambda: self._queue_sequence("JV"))
        btn_jt.clicked.connect(lambda: self._queue_sequence("JT"))
        form.addRow(btn_jv); form.addRow(btn_jt)

        dock.setWidget(host)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_hw_menu(self):
        m = self.menuBar().addMenu("&Hardware")
        m.addAction(QtGui.QAction("Connect Bias SM (VISA/Prologix)", self, triggered=self._connect_bias_sm))
        m.addAction(QtGui.QAction("Connect Dummy Bias SM", self, triggered=lambda: self._connect_dummy_sm(True)))
        m.addAction(QtGui.QAction("Connect Readout SM (VISA/Prologix)", self, triggered=self._connect_read_sm))
        m.addAction(QtGui.QAction("Connect Dummy Readout SM", self, triggered=lambda: self._connect_dummy_sm(False)))
        m.addSeparator()
        m.addAction(QtGui.QAction("Connect Switch Board", self, triggered=self._connect_switch))
        m.addAction(QtGui.QAction("Connect Dummy Switch", self, triggered=self._connect_dummy_switch))
        m.addSeparator(); m.addAction(QtGui.QAction("&Exit", self, triggered=self.close))

    # ------------------------------------------------------------------ HW connect helpers
    def _connect_bias_sm(self):
        port, ok = QtWidgets.QInputDialog.getText(self, "Bias SM", "VISA or Prologix:",
                                                  text="USB0::0x05E6::0x2450::INSTR")
        if not ok: return
        try:
            adp = VISAAdapter(port) if port.startswith("USB") else PrologixAdapter(port, 5)
            self.bias_sm = Keithley2400(adp); self.bias_sm.write("SYST:LANG SCPI")
            QtWidgets.QMessageBox.information(self, "Bias SM", "Connected.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Bias SM", f"Failed: {e}")

    def _connect_read_sm(self):
        port, ok = QtWidgets.QInputDialog.getText(self, "Readout SM", "VISA or Prologix:",
                                                  text="USB0::0x05E6::0x2400::INSTR")
        if not ok: return
        try:
            adp = VISAAdapter(port) if port.startswith("USB") else PrologixAdapter(port, 5)
            self.read_sm = ReadoutSafe2400(adp)
            QtWidgets.QMessageBox.information(self, "Readout SM", "Connected (0 V locked).")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Readout SM", f"Failed: {e}")

    def _connect_dummy_sm(self, is_bias: bool):
        sm = DummyKeithley2400()
        if is_bias:
            self.bias_sm = sm
            QtWidgets.QMessageBox.information(self, "Dummy Bias SM", "Dummy connected.")
        else:
            self.read_sm = sm
            QtWidgets.QMessageBox.information(self, "Dummy Readout SM", "Dummy connected.")

    def _connect_switch(self):
        port, ok = QtWidgets.QInputDialog.getText(self, "Switch Board", "Serial port:",
                                                  text="/dev/ttyACM0")
        if not ok: return
        try:
            self.switch = SwitchBoard(port)
            QtWidgets.QMessageBox.information(self, "Switch", "Connected.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Switch", f"Failed: {e}")

    def _connect_dummy_switch(self):
        self.switch = SwitchBoardDummy()
        QtWidgets.QMessageBox.information(self, "Dummy Switch", "Dummy connected.")

    def _send_spi(self, txt: str):
        if not self.switch: return
        try:
            b1, b2 = (int(x, 16) for x in txt.split())
            self.switch.spi_write(b1, b2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "SPI", f"Format “HH HH” hex – {e}")

    # ------------------------------------------------------------------ pixel & sequence
    def _on_pixel_clicked(self, idx: int):
        if self.switch: self.switch.route(idx)

        # update Inputs pane
        if hasattr(self.inputs, "pixel_index"):
            w = getattr(self.inputs, "pixel_index")
            if hasattr(w, "setValue"):
                w.blockSignals(True); w.setValue(idx); w.blockSignals(False)
            elif hasattr(w, "setText"):
                w.blockSignals(True); w.setText(str(idx)); w.blockSignals(False)
            if hasattr(w, "update_parameter"): w.update_parameter()

        if self.rb_measure.isChecked():
            self.queue()   # uses make_procedure()

    def _queue_sequence(self, mode: str):
        start, end = sorted((self.seq_start.value(), self.seq_end.value()))
        old_mode = self.inputs.measurement_mode.currentText()
        for p in range(start, end+1):
            self.inputs.measurement_mode.setCurrentText(mode)
            self.inputs.pixel_index.setValue(p)
            self.inputs.measurement_mode.update_parameter()
            self.inputs.pixel_index.update_parameter()
            self.queue()
        # restore mode
        self.inputs.measurement_mode.setCurrentText(old_mode)
        self.inputs.measurement_mode.update_parameter()

# ----------------------------------------------------------------------
def resource_path(rel_path):
    try: base = sys._MEIPASS
    except Exception: base = Path(__file__).parent
    return str(Path(base, rel_path))

# ----------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv); win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
