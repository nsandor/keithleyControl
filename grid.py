#!/usr/bin/env python3
# photodiode_array_live_grid.py  –  2025-07-17 (rev-E)
#
# Simplified photodiode array readout GUI **WITH HEATMAP + CSV EXPORT**
# • One Keithley sourcemeter (bias locked to 0 V)
# • USB 100:1 switch (Nano Every firmware v1.0) – or Dummy for offline testing
# • Reads the average current over N samples for each of the 100 pixels (single‑shot)
# • Displays results as a live‑updating 10 × 10 heat‑map (auto‑scales to current min/max)
# • Allows exporting the final 10 × 10 array to a CSV file (values in amperes)
# --- MODIFIED: Added IDN validation, status display, ACK-based sync,
# ---           bad channel reporting, and heatmap value display.
# --- MODIFIED (rev-D): Fixed plot update bug, added separate plot panels,
# ---                   plot titles, individual plot export, LED control,
# ---                   and autosave functionality with experiment naming.
# --- MODIFIED (rev-E): Reworked plot display to use a selector, added
# ---                   contextual plot settings, and made device
# ---                   communication tolerant to disconnections.
# ----------------------------------------------------------------------

import sys
import time
import logging
from pathlib import Path
from typing import Optional, List
from functools import partial

import numpy as np

# Optional deps: handle gracefully if missing
try:
    import pyvisa
except Exception:
    pyvisa = None

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None
# ----------------------------------------------------------------------
# Qt binding (PyQt5 preferred, fall back to PySide6)
# ----------------------------------------------------------------------
try:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
except ImportError:  # pragma: no cover – fallback
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

# ----------------------------------------------------------------------
# Matplotlib – embed in Qt and keep quiet
# ----------------------------------------------------------------------
import matplotlib

matplotlib.use("Qt5Agg")  # ensure Qt backend
matplotlib.set_loglevel("warning")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# noqa: E402 – after backend selection

# ----------------------------------------------------------------------
# Keithley instruments via PyMeasure – with a zero‑volt safety wrapper
# ----------------------------------------------------------------------
try:
    from pymeasure.instruments.keithley import Keithley2400
except ImportError:
    raise SystemExit("PyMeasure is required (pip install pymeasure)")


# ---------- helpers ----------
def _scan_visa_resources():
    items = []
    if not pyvisa:
        return items
    try:
        rm = pyvisa.ResourceManager()
        for r in rm.list_resources():
            label = r
            try:
                info = rm.resource_info(r)
                manu = getattr(info, "manufacturer_name", "") or ""
                model = getattr(info, "model_name", "") or ""
                serno = getattr(info, "serial_number", "") or ""
                alias = getattr(info, "alias", "") or ""
                parts = [p for p in [alias, manu, model, serno] if p]
                if parts:
                    label = f"{r} — " + " ".join(parts)
            except Exception:
                pass
            items.append({"kind": "visa", "value": r, "label": label})
    except Exception:
        pass
    return items


def _scan_serial_ports():
    items = []
    if not list_ports:
        return items
    try:
        for p in list_ports.comports():
            if p.description and "n/a" in p.description.lower():
                continue

            hints = []
            if p.manufacturer:
                hints.append(p.manufacturer)
            if p.product:
                hints.append(p.product)
            if p.description and p.description not in hints:
                hints.append(p.description)
            if p.vid and p.pid:
                hints.append(f"{p.vid:04x}:{p.pid:04x}")
            if p.serial_number:
                hints.append(f"SN:{p.serial_number}")
            label = f"{p.device} — " + " ".join(hints) if hints else p.device

            score = 0
            t = (p.manufacturer or "").lower() + " " + (p.product or "").lower()
            if any(
                k in t
                for k in [
                    "arduino",
                    "atmega",
                    "leonardo",
                    "nano every",
                    "megaavr",
                    "microchip",
                ]
            ):
                score += 10

            items.append(
                {"kind": "serial", "value": p.device, "label": label, "score": score}
            )
    except Exception:
        pass

    items.sort(key=lambda d: (-d.get("score", 0), d["label"]))
    return items


class _DevicePickDialog(QtWidgets.QDialog):
    def __init__(
        self, parent, title="Select device", show_gpib_addr=False, default_gpib_addr=5
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.combo = QtWidgets.QComboBox(self)
        self.combo.setMinimumWidth(520)
        self.refresh_btn = QtWidgets.QPushButton("Rescan")
        self.manual_edit = QtWidgets.QLineEdit(self)
        self.manual_edit.setPlaceholderText(
            "Or paste a resource/port (e.g., USB0::..., GPIB0::..., ASRL4::INSTR, COM3, /dev/ttyACM0, 192.168.1.50)"
        )
        self.manual_edit.setClearButtonEnabled(True)
        self.gpib_row = QtWidgets.QWidget(self)
        gpib_layout = QtWidgets.QHBoxLayout(self.gpib_row)
        gpib_layout.setContentsMargins(0, 0, 0, 0)
        self.gpib_label = QtWidgets.QLabel("GPIB address (Prologix):")
        self.gpib_spin = QtWidgets.QSpinBox(self)
        self.gpib_spin.setRange(0, 30)
        self.gpib_spin.setValue(default_gpib_addr)
        gpib_layout.addWidget(self.gpib_label)
        gpib_layout.addWidget(self.gpib_spin)
        gpib_layout.addStretch(1)
        self.gpib_row.setVisible(show_gpib_addr)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Detected devices:"))
        layout.addWidget(self.combo)
        layout.addWidget(self.refresh_btn)
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Advanced:"))
        layout.addWidget(self.manual_edit)
        layout.addWidget(self.gpib_row)
        layout.addWidget(btns)
        self._items = []
        self.refresh_btn.clicked.connect(self._populate)
        self._populate()

    def _populate(self):
        self.combo.clear()
        self._items = []
        visa_items = _scan_visa_resources()
        serial_items = _scan_serial_ports()
        if visa_items:
            self.combo.addItem("— VISA resources —")
            self.combo.model().item(self.combo.count() - 1).setEnabled(False)
            self._items.append({"kind": "header"})
            for it in visa_items:
                self.combo.addItem(it["label"])
                self._items.append(it)
        if serial_items:
            if visa_items:
                self.combo.addItem("— Serial ports —")
                self.combo.model().item(self.combo.count() - 1).setEnabled(False)
                self._items.append({"kind": "header"})
            for it in serial_items:
                self.combo.addItem(it["label"])
                self._items.append(it)
        if self.combo.count() == 0:
            self.combo.addItem("(none found) – use manual entry")

    def get_selection(self):
        manual = self.manual_edit.text().strip()
        if manual:
            return manual, (
                self.gpib_spin.value() if self.gpib_row.isVisible() else None
            )
        idx = self.combo.currentIndex()
        if idx < 0 or idx >= len(self._items):
            return None, None
        item = self._items[idx]
        if item.get("kind") in ("header", None):
            return None, None
        return item["value"], (
            self.gpib_spin.value() if self.gpib_row.isVisible() else None
        )


class ReadoutSafe2400(Keithley2400):
    """Keithley 24xx that *must* remain at 0 V source output."""

    def __init__(self, adapter, **kwargs):
        super().__init__(adapter, **kwargs)
        self.write("SOUR:FUNC VOLT")
        self.source_voltage = 0
        self.disable_source()

    @property
    def source_voltage(self):
        return float(self.ask(":SOUR:VOLT?").strip())

    @source_voltage.setter
    def source_voltage(self, val):
        if abs(val) > 1e-9:
            raise RuntimeError("Readout SM must remain at 0 V – refusing")
        super(ReadoutSafe2400, self.__class__).source_voltage.fset(self, 0)

    def enable_source(self):
        if abs(self.source_voltage) > 1e-9:
            raise RuntimeError("Refusing to enable source – voltage ≠ 0 V")
        super().enable_source()


class DummyKeithley2400:
    """Mimics *just enough* of a 2400 for development without hardware."""

    def __init__(self):
        self._current = 0.0
        np.random.seed(0)

    def measure_current(self, **_):
        pass

    def enable_source(self):
        pass

    def disable_source(self):
        pass

    def reset(self):
        pass

    @property
    def current(self):
        self._current = 50e-9 * (2 * np.random.rand() - 1)
        return self._current

    def ask(self, _):
        return str(self.current)


class SwitchBoard:
    """Simple wrapper for the 100:1 switch USB board, with ACK sync."""

    def __init__(self, port: str, baud: int = 9600, timeout: float = 2.0):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)

    def route(self, idx: int):
        if not 1 <= idx <= 100:
            raise ValueError("Pixel index must be 1‑100")
        self.ser.write(f"{idx}\n".encode())
        response = self.ser.readline()
        if b"ACK" not in response:
            raise TimeoutError(
                f"Switch did not ACK for pixel {idx}. Response: {response.decode(errors='ignore')}"
            )

    def close(self):
        self.ser.close()


class DummySwitchBoard:
    def route(self, *_):
        pass

    def close(self):
        pass


class ScanWorker(QtCore.QObject):
    """Runs in a separate thread – performs one full 1→100 scan."""

    pixelDone = QtCore.pyqtSignal(int, float)
    deviceError = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        sm,
        switch,
        n_samples: int,
        nplc: float,
        inter_sample_delay_s: float = 0.05,
    ):
        super().__init__()
        self._sm = sm
        self._sw = switch
        self._n = max(1, n_samples)
        self._nplc = max(0.01, nplc)
        self._inter_sample_delay = inter_sample_delay_s
        self._stop = False
        self._paused = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self._sm.reset()
            self._sm.enable_source()
            self._sm.measure_current(nplc=self._nplc, auto_range=False)
            self._sm.current_range = 1e-7
        except Exception as e:
            self.deviceError.emit(f"Failed to configure Sourcemeter: {e}")
            self.finished.emit()
            return

        for p in range(1, 101):
            while self._paused and not self._stop:
                QtCore.QThread.msleep(100)
            if self._stop:
                break

            try:
                self._sw.route(p)  # This now blocks until ACK is received
            except Exception as e:
                logging.warning(f"Switch route failed for pixel {p}: {e}")
                self.deviceError.emit(f"Switch connection lost: {e}")
                break

            vals = []
            for _ in range(self._n):
                if self._stop:
                    break
                try:
                    val = float(self._sm.current)
                    # This delay is for measurement settling, not switching
                    QtCore.QThread.msleep(int(self._inter_sample_delay * 1000))
                except Exception as e:
                    logging.warning(f"Read failed: {e}")
                    self.deviceError.emit(f"Sourcemeter connection lost: {e}")
                    val = np.nan
                    break  # Break inner loop
                vals.append(np.abs(val))

            if self._stop or np.isnan(vals).any():
                break

            self.pixelDone.emit(p, float(np.nanmean(vals)))

        try:
            self._sm.disable_source()
        except Exception:
            # Ignore errors on shutdown, as device may be disconnected
            pass
        self.finished.emit()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stop = True
        self._paused = False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photodiode Array – Live Current Heat-map")
        self.resize(1000, 850)
        self.setStatusBar(QtWidgets.QStatusBar(self))

        self.data = np.full((10, 10), np.nan)
        self.sm: Optional[Keithley2400] = DummyKeithley2400()
        self.switch: Optional[SwitchBoard] = DummySwitchBoard()

        # --- Main Layout: Control Panel | Plot Area ---
        central = QtWidgets.QSplitter()
        self.setCentralWidget(central)
        ctrl_container = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_container)
        central.addWidget(ctrl_container)

        # --- Control Panel ---
        scan_settings_box = QtWidgets.QGroupBox("Scan Settings")
        scan_settings_layout = QtWidgets.QFormLayout(scan_settings_box)
        self.spin_nplc = QtWidgets.QDoubleSpinBox()
        self.spin_nplc.setDecimals(2)
        self.spin_nplc.setRange(0.01, 25)
        self.spin_nplc.setValue(1.0)
        scan_settings_layout.addRow("NPLC", self.spin_nplc)
        self.spin_nsamp = QtWidgets.QSpinBox()
        self.spin_nsamp.setRange(1, 100)
        self.spin_nsamp.setValue(5)
        scan_settings_layout.addRow("Samples / pixel", self.spin_nsamp)
        ctrl_layout.addWidget(scan_settings_box)

        self.btn_run_abort = QtWidgets.QPushButton("Run Scan")
        self.btn_pause_resume = QtWidgets.QPushButton("Pause")
        self.btn_pause_resume.setEnabled(False)
        run_layout = QtWidgets.QHBoxLayout()
        run_layout.addWidget(self.btn_run_abort)
        run_layout.addWidget(self.btn_pause_resume)
        ctrl_layout.addLayout(run_layout)

        self.btn_export = QtWidgets.QPushButton("Export All Data (CSV)…")
        self.btn_export.setEnabled(False)
        ctrl_layout.addWidget(self.btn_export)

        hw_box = QtWidgets.QGroupBox("Hardware Control")
        hw_layout = QtWidgets.QVBoxLayout(hw_box)
        h_hw = QtWidgets.QHBoxLayout()
        self.btn_connect_sm = QtWidgets.QPushButton("Connect SM…")
        self.btn_connect_sw = QtWidgets.QPushButton("Connect Switch…")
        self.btn_leds = QtWidgets.QPushButton("LEDs")
        self.btn_leds.setCheckable(True)
        self.btn_leds.setEnabled(False)
        h_hw.addWidget(self.btn_connect_sm)
        h_hw.addWidget(self.btn_connect_sw)
        h_hw.addWidget(self.btn_leds)
        hw_layout.addLayout(h_hw)
        status_box = QtWidgets.QGroupBox("Instrument Status")
        status_layout = QtWidgets.QVBoxLayout(status_box)
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(60)
        status_layout.addWidget(self.status_text)
        hw_layout.addWidget(status_box)
        sw_status_box = QtWidgets.QGroupBox("Inactive Switch Channels")
        sw_status_layout = QtWidgets.QVBoxLayout(sw_status_box)
        self.inactive_channels_display = QtWidgets.QLineEdit()
        self.inactive_channels_display.setReadOnly(True)
        sw_status_layout.addWidget(self.inactive_channels_display)
        hw_layout.addWidget(sw_status_box)
        ctrl_layout.addWidget(hw_box)

        # --- Output & Saving Controls ---
        save_box = QtWidgets.QGroupBox("Output & Saving")
        save_layout = QtWidgets.QFormLayout(save_box)
        self.edit_exp_name = QtWidgets.QLineEdit("MyExperiment")
        save_layout.addRow("Experiment Name:", self.edit_exp_name)
        h_folder = QtWidgets.QHBoxLayout()
        self.edit_output_folder = QtWidgets.QLineEdit(str(Path.home()))
        self.edit_output_folder.setReadOnly(True)
        self.btn_browse_folder = QtWidgets.QPushButton("Browse…")
        h_folder.addWidget(self.edit_output_folder)
        h_folder.addWidget(self.btn_browse_folder)
        save_layout.addRow("Output Folder:", h_folder)
        self.output_folder = Path(self.edit_output_folder.text())
        self.check_autosave = QtWidgets.QCheckBox("Autosave after scan")
        self.check_autosave.setChecked(True)
        save_layout.addRow(self.check_autosave)
        ctrl_layout.addWidget(save_box)

        # --- Plotting Controls (in a Stacked Widget) ---
        self.plot_settings_stack = QtWidgets.QStackedWidget()
        self._create_heatmap_settings_panel()
        self._create_histogram_settings_panel()
        ctrl_layout.addWidget(self.plot_settings_stack)

        ctrl_layout.addStretch(1)

        # --- Plot Area: Selector + Stacked Widget ---
        plot_area_widget = QtWidgets.QWidget()
        plot_area_layout = QtWidgets.QVBoxLayout(plot_area_widget)
        self.plot_selector = QtWidgets.QComboBox()
        self.plot_selector.addItems(["Heatmap", "Histogram"])
        plot_area_layout.addWidget(self.plot_selector)

        self.plot_stack = QtWidgets.QStackedWidget()
        plot_area_layout.addWidget(self.plot_stack)

        # Heatmap Figure
        self.figure_heatmap = plt.figure()
        self.canvas_heatmap = FigureCanvas(self.figure_heatmap)
        self.ax_heatmap = self.figure_heatmap.add_subplot(111)
        self.im = self.ax_heatmap.imshow(
            np.full((10, 10), np.nan),
            cmap="inferno",
            norm=LogNorm(vmin=1e-10, vmax=1e-7),
        )
        self.ax_heatmap.set_xticks([])
        self.ax_heatmap.set_yticks([])
        self.cbar = self.figure_heatmap.colorbar(
            self.im, ax=self.ax_heatmap, fraction=0.046, pad=0.04, label="Current (A)"
        )
        self.figure_heatmap.tight_layout(pad=2.5)
        self.plot_stack.addWidget(self.canvas_heatmap)

        # Histogram Figure
        self.figure_hist = plt.figure()
        self.canvas_hist = FigureCanvas(self.figure_hist)
        self.ax_hist = self.figure_hist.add_subplot(111)
        self.ax_hist.set_xlabel("Current (A)")
        self.ax_hist.set_ylabel("Pixel Count")
        self.ax_hist.grid(True, linestyle="--", alpha=0.6)
        self.figure_hist.tight_layout(pad=2.5)
        self.plot_stack.addWidget(self.canvas_hist)

        central.addWidget(plot_area_widget)
        central.setStretchFactor(1, 1)

        # --- Connections ---
        self.btn_run_abort.clicked.connect(self.on_run_abort_clicked)
        self.btn_pause_resume.clicked.connect(self.on_pause_resume_clicked)
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_connect_sm.clicked.connect(self._connect_sm)
        self.btn_connect_sw.clicked.connect(self._connect_switch)
        self.btn_leds.toggled.connect(self._on_led_toggled)
        self.btn_browse_folder.clicked.connect(self._select_output_folder)
        self.plot_selector.currentIndexChanged.connect(self._on_plot_selected)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[ScanWorker] = None
        self._is_paused = False
        self.sm_idn = "Sourcemeter: (not connected)"
        self.switch_idn = "Switch: (not connected)"
        self.inactive_channels: List[int] = []
        self.value_text_annotations: List = []
        self.bad_channel_markers = None
        self._update_status_text()
        self._on_plot_selected(0)  # Initialize to heatmap view
        self._update_plots()

    def _create_heatmap_settings_panel(self):
        plot_box = QtWidgets.QGroupBox("Heatmap Settings")
        plot_layout = QtWidgets.QFormLayout(plot_box)
        self.edit_heatmap_title = QtWidgets.QLineEdit("Photodiode Current")
        plot_layout.addRow("Title:", self.edit_heatmap_title)
        self.combo_colormap = QtWidgets.QComboBox()
        self.colormaps = [
            "inferno",
            "viridis",
            "plasma",
            "magma",
            "cividis",
            "gray_r",
            "jet",
        ]
        self.combo_colormap.addItems(self.colormaps)
        plot_layout.addRow("Colormap:", self.combo_colormap)
        self.check_log_scale_heatmap = QtWidgets.QCheckBox("Logarithmic Color Scale")
        self.check_log_scale_heatmap.setChecked(True)
        plot_layout.addRow(self.check_log_scale_heatmap)
        self.check_auto_scale = QtWidgets.QCheckBox("Auto-scale Color Limit")
        self.check_auto_scale.setChecked(True)
        plot_layout.addRow(self.check_auto_scale)
        self.edit_vmin = QtWidgets.QLineEdit("1e-10")
        self.edit_vmax = QtWidgets.QLineEdit("1e-7")
        self.edit_vmin.setEnabled(False)
        self.edit_vmax.setEnabled(False)
        minmax_layout = QtWidgets.QHBoxLayout()
        minmax_layout.addWidget(QtWidgets.QLabel("Min:"))
        minmax_layout.addWidget(self.edit_vmin)
        minmax_layout.addWidget(QtWidgets.QLabel("Max:"))
        minmax_layout.addWidget(self.edit_vmax)
        plot_layout.addRow("Manual Limits:", minmax_layout)
        self.check_show_values = QtWidgets.QCheckBox("Show Pixel Values")
        plot_layout.addRow(self.check_show_values)
        self.btn_export_heatmap = QtWidgets.QPushButton("Export Heatmap PNG…")
        plot_layout.addRow(self.btn_export_heatmap)
        self.plot_settings_stack.addWidget(plot_box)

        # Connections
        self.edit_heatmap_title.editingFinished.connect(partial(self._update_plots))
        self.combo_colormap.currentIndexChanged.connect(partial(self._update_plots))
        self.check_log_scale_heatmap.stateChanged.connect(partial(self._update_plots))
        self.check_show_values.stateChanged.connect(partial(self._update_plots))
        self.edit_vmin.editingFinished.connect(partial(self._update_plots))
        self.edit_vmax.editingFinished.connect(partial(self._update_plots))
        self.check_auto_scale.toggled.connect(self._update_plot_controls_state)
        self.btn_export_heatmap.clicked.connect(self._export_heatmap)

    def _create_histogram_settings_panel(self):
        hist_box = QtWidgets.QGroupBox("Histogram Settings")
        hist_layout = QtWidgets.QFormLayout(hist_box)
        self.edit_hist_title = QtWidgets.QLineEdit("Current Distribution")
        hist_layout.addRow("Title:", self.edit_hist_title)
        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setRange(5, 200)
        self.spin_bins.setValue(25)
        hist_layout.addRow("Number of Bins:", self.spin_bins)
        self.check_log_scale_hist = QtWidgets.QCheckBox("Logarithmic Y-Axis")
        hist_layout.addRow(self.check_log_scale_hist)
        self.btn_export_hist = QtWidgets.QPushButton("Export Histogram PNG…")
        hist_layout.addRow(self.btn_export_hist)
        self.plot_settings_stack.addWidget(hist_box)

        # Connections
        self.edit_hist_title.editingFinished.connect(partial(self._update_plots))
        self.spin_bins.valueChanged.connect(partial(self._update_plots))
        self.check_log_scale_hist.stateChanged.connect(partial(self._update_plots))
        self.btn_export_hist.clicked.connect(self._export_histogram)

    def _on_plot_selected(self, index: int):
        self.plot_stack.setCurrentIndex(index)
        self.plot_settings_stack.setCurrentIndex(index)

    def on_run_abort_clicked(self):
        if self._thread is not None:
            self._abort_scan()
        else:
            self._start_scan()

    def on_pause_resume_clicked(self):
        if self._worker is None:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._worker.pause()
            self.btn_pause_resume.setText("Resume")
        else:
            self._worker.resume()
            self.btn_pause_resume.setText("Pause")

    def _start_scan(self):
        self.data.fill(np.nan)
        self._update_plots(reset=True)
        self._worker = ScanWorker(
            self.sm,
            self.switch,
            n_samples=self.spin_nsamp.value(),
            nplc=self.spin_nplc.value(),
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._worker.pixelDone.connect(self._update_pixel)
        self._worker.deviceError.connect(self._handle_device_error)
        self._worker.finished.connect(self._scan_finished)
        self._thread.started.connect(self._worker.run)
        self.btn_run_abort.setText("Abort")
        self.btn_pause_resume.setText("Pause")
        self.btn_pause_resume.setEnabled(True)
        self.btn_export.setEnabled(False)
        self._is_paused = False
        self._thread.start()

    def _abort_scan(self):
        if self._worker:
            self.btn_run_abort.setText("Aborting…")
            self.btn_run_abort.setEnabled(False)
            self._worker.stop()

    def _scan_finished(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self.btn_run_abort.setText("Run Scan")
        self.btn_run_abort.setEnabled(True)
        self.btn_pause_resume.setText("Pause")
        self.btn_pause_resume.setEnabled(False)
        self.btn_export.setEnabled(True)
        self._thread = None
        self._worker = None
        if not np.all(np.isnan(self.data)):  # Only autosave if some data was collected
            self._autosave_results()

    def _handle_device_error(self, message: str):
        logging.error(f"Device error during scan: {message}")
        # Stop the worker. The finished signal will call _scan_finished for cleanup.
        if self._worker:
            self._worker.stop()

        QtWidgets.QMessageBox.critical(
            self,
            "Device Error",
            f"A device has disconnected or failed.\n\n{message}\n\nThe scan has been aborted.",
        )

        # Reset the specific device that failed
        if "switch" in message.lower():
            self.switch = DummySwitchBoard()
            self.switch_idn = "Switch: (disconnected)"
            self.btn_connect_sw.setStyleSheet("")
            self.btn_leds.setEnabled(False)
        if "sourcemeter" in message.lower():
            self.sm = DummyKeithley2400()
            self.sm_idn = "Sourcemeter: (disconnected)"
            self.btn_connect_sm.setStyleSheet("")

        self._update_status_text()

    def _update_pixel(self, idx: int, i_avg: float):
        r, c = divmod(idx - 1, 10)
        self.data[r, c] = i_avg
        self._update_plots()

    def _update_plots(self, reset: bool = False):
        # --- Update Heatmap Data ---
        self.ax_heatmap.set_title(self.edit_heatmap_title.text())
        for txt in self.value_text_annotations:
            txt.remove()
        self.value_text_annotations.clear()
        if self.bad_channel_markers:
            self.bad_channel_markers.remove()
            self.bad_channel_markers = None

        if reset:
            self.im.set_data(np.full((10, 10), np.nan))
        else:
            valid_data = self.data[~np.isnan(self.data)]
            if valid_data.size == 0:
                self.im.set_data(np.full((10, 10), np.nan))
            else:
                self.im.set_data(self.data)
                use_log = self.check_log_scale_heatmap.isChecked()
                if self.check_auto_scale.isChecked():
                    vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
                    if vmin == vmax:
                        vmin = max(1e-12, vmin * 0.9)
                        vmax = max(1e-12, vmax * 1.1)
                    if vmin <= 0 and use_log:
                        vmin = 1e-12  # Log scale can't handle zero or negative
                    self.edit_vmin.setText(f"{vmin:.3e}")
                    self.edit_vmax.setText(f"{vmax:.3e}")
                else:
                    try:
                        vmin, vmax = float(self.edit_vmin.text()), float(
                            self.edit_vmax.text()
                        )
                    except ValueError:
                        vmin, vmax = 1e-10, 1e-7

                self.im.set_cmap(self.combo_colormap.currentText())
                if use_log:
                    self.im.set_norm(
                        LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-11))
                    )
                else:
                    self.im.set_norm(Normalize(vmin=vmin, vmax=vmax))

                if self.check_show_values.isChecked():
                    norm = self.im.norm
                    for r in range(10):
                        for c in range(10):
                            val = self.data[r, c]
                            if not np.isnan(val):
                                norm_val = norm(val) if norm and val > 0 else 0.5
                                color = "white" if norm_val < 0.5 else "black"
                                self.value_text_annotations.append(
                                    self.ax_heatmap.text(
                                        c,
                                        r,
                                        f"{val:.2e}",
                                        ha="center",
                                        va="center",
                                        color=color,
                                        fontsize=8,
                                    )
                                )

        if self.inactive_channels:
            rows, cols = divmod(np.array(self.inactive_channels) - 1, 10)
            self.bad_channel_markers = self.ax_heatmap.scatter(
                cols, rows, marker="x", color="red", s=50, linewidths=1.5
            )

        self.figure_heatmap.tight_layout(pad=2.5)
        self.canvas_heatmap.draw_idle()

        # --- Update Histogram Data ---
        self.ax_hist.clear()
        self.ax_hist.set_title(self.edit_hist_title.text())
        valid_data = self.data[~np.isnan(self.data)]
        if valid_data.size > 0:
            self.ax_hist.hist(
                valid_data.flatten(), bins=self.spin_bins.value(), color="gray"
            )
        if self.check_log_scale_hist.isChecked():
            self.ax_hist.set_yscale("log")
        self.ax_hist.grid(True, linestyle="--", alpha=0.6)
        self.ax_hist.set_xlabel("Current (A)")
        self.ax_hist.set_ylabel("Pixel Count")

        self.figure_hist.tight_layout(pad=2.5)
        self.canvas_hist.draw_idle()

    def _update_plot_controls_state(self):
        is_manual = not self.check_auto_scale.isChecked()
        self.edit_vmin.setEnabled(is_manual)
        self.edit_vmax.setEnabled(is_manual)
        self._update_plots()

    def _update_status_text(self):
        self.status_text.setText(f"{self.sm_idn}\n{self.switch_idn}")

    def _export_csv(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            str(self.output_folder / "photodiode_scan.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not fname:
            return
        try:
            np.savetxt(fname, self.data, delimiter=",", fmt="%.5e")
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed: {e}")

    def _export_figure(self, figure, default_name):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            str(self.output_folder / default_name),
            "PNG files (*.png);;All files (*)",
        )
        if not fname:
            return
        try:
            figure.savefig(fname, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed: {e}")

    def _export_heatmap(self):
        self._export_figure(self.figure_heatmap, "heatmap.png")

    def _export_histogram(self):
        self._export_figure(self.figure_hist, "histogram.png")

    def _select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(self.output_folder)
        )
        if folder:
            self.output_folder = Path(folder)
            self.edit_output_folder.setText(str(self.output_folder))

    def _autosave_results(self):
        if not self.check_autosave.isChecked():
            return
        exp_name = self.edit_exp_name.text().strip()
        if not exp_name:
            QtWidgets.QMessageBox.warning(
                self, "Autosave", "Please enter an experiment name to autosave."
            )
            return
        if not self.output_folder or not self.output_folder.is_dir():
            QtWidgets.QMessageBox.warning(
                self, "Autosave", "Please select a valid output folder to autosave."
            )
            return

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_folder_name = f"{exp_name}_{timestamp}"
        run_folder = self.output_folder / run_folder_name

        try:
            run_folder.mkdir(parents=True, exist_ok=True)
            np.savetxt(run_folder / "data.csv", self.data, delimiter=",", fmt="%.5e")
            self.figure_heatmap.savefig(
                run_folder / "heatmap.png", dpi=300, bbox_inches="tight"
            )
            self.figure_hist.savefig(
                run_folder / "histogram.png", dpi=300, bbox_inches="tight"
            )
            self.statusBar().showMessage(f"Autosaved to {run_folder}", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Autosave Error", f"Failed to save results: {e}"
            )

    def _on_led_toggled(self, checked: bool):
        cmd = b"LED ON\n" if checked else b"LED OFF\n"
        try:
            if self.switch and hasattr(self.switch, "ser"):
                self.switch.ser.write(cmd)
                self.btn_leds.setStyleSheet(
                    "background-color: lightgreen" if checked else ""
                )
            else:
                raise IOError("Switch not connected or is a dummy device.")
        except Exception as e:
            logging.error(f"Failed to send LED command: {e}")
            self.btn_leds.setChecked(not checked)
            self.btn_leds.setStyleSheet("background-color: red")
            QtWidgets.QMessageBox.warning(
                self, "LED Control", f"Failed to send command: {e}"
            )

    def _connect_sm(self):
        dlg = _DevicePickDialog(self, title="Connect Sourcemeter", show_gpib_addr=True)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        selection, gpib_addr = dlg.get_selection()
        if not selection:
            return

        try:
            from pymeasure.adapters import VISAAdapter, PrologixAdapter

            res = selection.strip()
            up = res.upper()
            use_visa = up.startswith(("USB", "GPIB", "TCPIP")) and pyvisa
            if use_visa:
                adapter = VISAAdapter(res)
            else:
                adapter = PrologixAdapter(res, gpib_addr or 5, gpib_read_timeout=3000)
            adapter.connection.timeout = 20000
            adapter.write("++mode 1")  # controller
            adapter.write("++auto 0")  # *crucial* – we will read explicitly
            adapter.write("++eoi 1")  # assert EOI with last byte
            self.sm = ReadoutSafe2400(adapter)
            ident = self.sm.adapter.ask("*IDN?")
            if not (ident and len(ident) > 3):
                raise RuntimeError("Device did not respond to *IDN?")
            if use_visa:
                self.sm.use_rear_terminals()
                # means we are using the 2400, make sure to use the back panel

            self.sm_idn = f"Sourcemeter: {ident.strip()}"
            self._update_status_text()
            self.btn_connect_sm.setStyleSheet("background-color: lightgreen")
            QtWidgets.QMessageBox.information(
                self, "Sourcemeter", f"Connected: {ident}\nLocked at 0 V."
            )
        except Exception as e:
            self.sm = DummyKeithley2400()
            self.sm_idn = "Sourcemeter: DUMMY TEST DEVICE"
            self._update_status_text()
            self.btn_connect_sm.setStyleSheet("")
            QtWidgets.QMessageBox.critical(
                self, "Sourcemeter", f"Failed to connect/validate: {e}"
            )

    def _connect_switch(self):
        dlg = _DevicePickDialog(
            self, title="Connect Switch Board", show_gpib_addr=False
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        port, _ = dlg.get_selection()
        if not port:
            return

        try:
            self.switch = SwitchBoard(port)
            ser = self.switch.ser
            ser.write(b"VERBOSE OFF\n")
            ser.readline()
            ser.write(b"IDN\n")
            idn_response = ser.readline().decode("utf-8").strip()
            if not idn_response:
                raise RuntimeError("Switch did not respond to IDN.")

            self.switch_idn = f"Switch: {idn_response}"
            self.btn_connect_sw.setStyleSheet("background-color: lightgreen")
            self.btn_leds.setEnabled(True)

            ser.write(b"SWSTATUS\n")
            status_response = ser.readline().decode("utf-8").strip()
            if status_response:
                self.inactive_channels = [
                    int(x) for x in status_response.split(",") if x.strip()
                ]
                self.inactive_channels_display.setText(status_response)
            else:
                self.inactive_channels = []
                self.inactive_channels_display.setText("(none)")

            self._update_status_text()
            self._update_plots(reset=True)  # Redraw to show inactive channels
            QtWidgets.QMessageBox.information(
                self, "Switch", f"Connected on {port}.\nID: {idn_response}"
            )

        except Exception as e:
            self.switch = DummySwitchBoard()
            self.switch_idn = "Switch: (not connected)"
            self.inactive_channels = []
            self.inactive_channels_display.clear()
            self._update_status_text()
            self.btn_connect_sw.setStyleSheet("")
            self.btn_leds.setEnabled(False)
            self._update_plots(reset=True)
            QtWidgets.QMessageBox.critical(self, "Switch", f"Failed: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
