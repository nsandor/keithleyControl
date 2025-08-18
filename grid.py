#!/usr/bin/env python3
# photodiode_array_live_grid.py  –  2025‑07‑17 (rev‑A)
#
# Simplified photodiode array readout GUI **WITH HEATMAP + CSV EXPORT**
# • One Keithley sourcemeter (bias locked to 0 V)
# • USB 100:1 switch (Nano Every firmware v1.0) – or Dummy for offline testing
# • Reads the average current over N samples for each of the 100 pixels (single‑shot)
# • Displays results as a live‑updating 10 × 10 heat‑map (auto‑scales to current min/max)
# • Allows exporting the final 10 × 10 array to a CSV file (values in amperes)
# ----------------------------------------------------------------------

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------
# Qt binding (PyQt5 preferred, fall back to PySide6)
# ----------------------------------------------------------------------
try:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
except ImportError:  # pragma: no cover – fallback
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

# ----------------------------------------------------------------------
# Matplotlib – embed in Qt and keep quiet
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Qt5Agg")  # ensure Qt backend
matplotlib.set_loglevel("warning")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# noqa: E402 – after backend selection

# ----------------------------------------------------------------------
# Keithley instruments via PyMeasure – with a zero‑volt safety wrapper
# ----------------------------------------------------------------------
try:
    from pymeasure.instruments.keithley import Keithley2400
except ImportError:
    raise SystemExit("PyMeasure is required (pip install pymeasure)")


class ReadoutSafe2400(Keithley2400):
    """Keithley 24xx that *must* remain at 0 V source output.

    Any attempt to change the source to a non‑zero value raises RuntimeError.
    """

    def __init__(self, adapter, **kwargs):
        super().__init__(adapter, **kwargs)
        self.write("SOUR:FUNC VOLT")
        self.source_voltage = 0  # enforce immediately
        self.disable_source()

    # NB: PyMeasure already exposes .source_voltage property – we override setter
    @property  # type: ignore[override]
    def source_voltage(self):  # noqa: D401 – getter
        return float(self.ask(":SOUR:VOLT?").strip())

    @source_voltage.setter  # type: ignore[override]
    def source_voltage(self, val):
        if abs(val) > 1e-9:
            raise RuntimeError("Readout SM must remain at 0 V – refusing")
        super(ReadoutSafe2400, self.__class__).source_voltage.fset(self, 0)

    # lock enable as well
    def enable_source(self):
        # permit enabling only if we are really at 0 V
        if abs(self.source_voltage) > 1e-9:
            raise RuntimeError("Refusing to enable source – voltage ≠ 0 V")
        super().enable_source()


# ----------------------------------------------------------------------
# Dummy Keithley substitute (for offline testing)
# ----------------------------------------------------------------------
class DummyKeithley2400:
    """Mimics *just enough* of a 2400 for development without hardware."""

    def __init__(self):
        self._current = 0.0
        np.random.seed(0)

    # Measurement config stubs
    def measure_current(self, **_):
        pass

    # Source control stubs
    def enable_source(self):
        pass

    def disable_source(self):
        pass

    # Query interface
    @property
    def current(self):  # noqa: D401 getter only
        # Return deterministic fake current with noise [‑50 nA, +50 nA]
        self._current = 50e-9 * (2 * np.random.rand() - 1)
        return self._current

    def ask(self, _):
        return str(self.current)


# ----------------------------------------------------------------------
# Switch‑board serial wrapper (Nano Every firmware v1.0)
# ----------------------------------------------------------------------
try:
    import serial

    class SwitchBoard:
        """Simple *route‑only* wrapper for the 100:1 switch USB board."""

        def __init__(self, port: str, baud: int = 9600, timeout: float = 1):
            self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)

        def route(self, idx: int):
            if not 1 <= idx <= 100:
                raise ValueError("Pixel index must be 1‑100")
            self.ser.write(f"{idx}\n".encode())
            # ignore response – assume firmware echoes OK

        def close(self):
            self.ser.close()

except ImportError:  # pragma: no cover – serial missing is OK for dummy work

    class SwitchBoard:  # type: ignore
        def __init__(self, *_, **__):
            pass

        def route(self, *_):
            pass

        def close(self):
            pass


class DummySwitchBoard:
    def route(self, *_):
        pass

    def close(self):
        pass


# ----------------------------------------------------------------------
# Worker: scans pixels (1→100) ONCE and emits averaged current values
# ----------------------------------------------------------------------
class ScanWorker(QtCore.QObject):
    """Runs in a separate thread – performs one full 1→100 scan."""

    pixelDone = QtCore.pyqtSignal(int, float)  # pixel index, average current (A)
    finished = QtCore.pyqtSignal()

    def __init__(self, sm, switch, n_samples: int, nplc: float, delay_s: float = 0.5):
        super().__init__()
        self._sm = sm
        self._sw = switch
        self._n = max(1, n_samples)
        self._nplc = max(0.01, nplc)
        self._delay = delay_s
        self._stop = False

    @QtCore.pyqtSlot()
    def run(self):
        self._sm.reset()
        self._sm.enable_source()
        
        # Configure instrument once
        try:
            self._sm.measure_current(nplc=self._nplc,auto_range=False)
            self._sm.current_range = 1e-7
        except Exception:
            # Dummy – silently ignore
            pass

        # Loop over 100 pixels (single‑shot)
        for p in range(1, 101):
            if self._stop:
                break
            try:
                self._sw.route(p)
            except Exception as e:
                logging.warning(f"Switch route failed for pixel {p}: {e}")
            QtCore.QThread.msleep(int(self._delay * 1000))

            vals = []
            for _ in range(self._n):
                try:
                    val = float(self._sm.current)
                    QtCore.QThread.msleep(int(self._delay * 1000))
                except Exception as e:
                    logging.warning(f"Read failed: {e}")
                    val = np.nan
                vals.append(np.abs(val))
            avg = float(np.nanmean(vals))
            self.pixelDone.emit(p, avg)
        self._sm.disable_source()
        self.finished.emit()

    def stop(self):
        self._stop = True


# ----------------------------------------------------------------------
# Main application window
# ----------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photodiode Array – Live Current Heat‑map")
        self.resize(900, 600)

        # Data store (10×10) initialised to NaN
        self.data = np.full((10, 10), np.nan)

        # Hardware handles (default to dummy)
        self.sm: Optional[Keithley2400] = DummyKeithley2400()
        self.switch: Optional[SwitchBoard] = DummySwitchBoard()

        # UI setup – splitter: left (controls) | right (heat‑map)
        central = QtWidgets.QSplitter()
        self.setCentralWidget(central)

        # -------------------------------------------------- Control panel
        ctrl = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QFormLayout(ctrl)
        central.addWidget(ctrl)

        # NPLC spin box
        self.spin_nplc = QtWidgets.QDoubleSpinBox()
        self.spin_nplc.setDecimals(2)
        self.spin_nplc.setRange(0.01, 25)
        self.spin_nplc.setValue(1.0)
        ctrl_layout.addRow("NPLC", self.spin_nplc)

        # N‑samples spin box
        self.spin_nsamp = QtWidgets.QSpinBox()
        self.spin_nsamp.setRange(1, 100)
        self.spin_nsamp.setValue(5)
        ctrl_layout.addRow("Samples / pixel", self.spin_nsamp)

        # Start / Export buttons
        btn_start = QtWidgets.QPushButton("Run Scan")
        btn_export = QtWidgets.QPushButton("Export CSV…")
        btn_export.setEnabled(False)
        ctrl_layout.addRow(btn_start, btn_export)

        # Hardware connect buttons
        h_hw = QtWidgets.QHBoxLayout()
        btn_sm = QtWidgets.QPushButton("Connect SM…")
        btn_sw = QtWidgets.QPushButton("Connect Switch…")
        h_hw.addWidget(btn_sm)
        h_hw.addWidget(btn_sw)
        ctrl_layout.addRow(h_hw)

        ctrl_layout.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        )

        # -------------------------------------------------- Heat‑map canvas
        self.figure = plt.figure(figsize=(5, 5))
        self.ax = self.figure.add_subplot(111)
        # Initial image (blank)
        self.im = self.ax.imshow(np.zeros((10, 10)), cmap="inferno", norm=LogNorm(vmin=1e-10, vmax=1e-7))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.cbar = self.figure.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04,
                                         label="Current (A)")
        self.canvas = FigureCanvas(self.figure)
        central.addWidget(self.canvas)
        central.setStretchFactor(1, 1)  # give heat‑map more space

        # -------------------------------------------------- Connections
        btn_start.clicked.connect(lambda: self._start_scan(btn_start, btn_export))
        btn_export.clicked.connect(self._export_csv)
        btn_sm.clicked.connect(self._connect_sm)
        btn_sw.clicked.connect(self._connect_switch)

        # Thread placeholder
        self._thread: Optional[QtCore.QThread] = None

    # -------------------------------------------------- Measurement logic
    def _start_scan(self, b_start, b_export):
        if self._thread is not None:
            return  # already running
        # Reset data store & heat‑map
        self.data.fill(np.nan)
        self._update_heatmap(force=True)

        # Spawn worker (single‑shot)
        worker = ScanWorker(
            self.sm,
            self.switch,
            n_samples=self.spin_nsamp.value(),
            nplc=self.spin_nplc.value(),
        )
        th = QtCore.QThread()
        worker.moveToThread(th)
        worker.pixelDone.connect(self._update_pixel)
        worker.finished.connect(th.quit)
        worker.finished.connect(lambda: self._scan_finished(b_start, b_export))
        th.started.connect(worker.run)
        th.finished.connect(worker.deleteLater)
        th.finished.connect(lambda: setattr(self, "_thread", None))
        # Keep reference
        self._thread = th
        self._worker = worker  # type: ignore[attr-defined]
        # UI state
        b_start.setEnabled(False)
        b_export.setEnabled(False)
        # Launch
        th.start()

    def _scan_finished(self, b_start, b_export):
        b_start.setEnabled(True)
        b_export.setEnabled(True)

    def _update_pixel(self, idx: int, i_avg: float):
        r, c = divmod(idx - 1, 10)
        self.data[r, c] = i_avg
        self._update_heatmap()

    def _update_heatmap(self, force: bool = False):
        if force:
            # set dummy range until real data arrive
            dummy_data = np.zeros((10, 10))
            self.im.set_data(dummy_data)
            #self.im.set_clim(0, 1)
            self.canvas.draw_idle()
            return
    
        # Only update when at least one value is valid
        if self.data is None or np.all(np.isnan(self.data)):
            return
    
        self.im.set_data(self.data)
    
        self.canvas.draw_idle()

    # -------------------------------------------------- CSV export
    def _export_csv(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "photodiode_scan.csv", "CSV files (*.csv);;All files (*)"
        )
        if not fname:
            return
        try:
            # Save in amperes; format 1.5e‑12 etc.
            np.savetxt(fname, self.data, delimiter=",", fmt="%.5e")
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed: {e}")

    # -------------------------------------------------- Hardware connect helpers
    def _connect_sm(self):
        port, ok = QtWidgets.QInputDialog.getText(
            self, "Sourcemeter", "VISA resource or IP:", text="ASRL4::INSTR"
        )
        if not ok or not port:
            return
        try:
            from pymeasure.adapters import VISAAdapter, PrologixAdapter

            adapter = (
                VISAAdapter(port)
                if port.upper().startswith("USB")
                else PrologixAdapter(port, 5,gpib_read_timeout=3000)
            )
            adapter.connection.timeout = 20000
            adapter.write('++mode 1')         # controller
            adapter.write('++auto 0')         # *crucial* – we will read explicitly
            adapter.write('++eoi 1')          # assert EOI with last byte
            self.sm = ReadoutSafe2400(adapter)
            QtWidgets.QMessageBox.information(self, "Sourcemeter", "Connected and locked at 0 V.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Sourcemeter", f"Failed: {e}")
            self.sm = DummyKeithley2400()

    def _connect_switch(self):
        port, ok = QtWidgets.QInputDialog.getText(
            self, "Switch Board", "Serial port:", text="COM3"
        )
        if not ok or not port:
            return
        try:
            self.switch = SwitchBoard(port)
            QtWidgets.QMessageBox.information(self, "Switch", "Connected.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Switch", f"Failed: {e}")
            self.switch = DummySwitchBoard()


# ----------------------------------------------------------------------
# Convenience for frozen executables (PyInstaller) – resource path helper
# ----------------------------------------------------------------------

def resource_path(rel_path):
    try:
        base = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base = Path(__file__).parent
    return str(Path(base, rel_path))


# ----------------------------------------------------------------------
# Application entry‑point
# ----------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
