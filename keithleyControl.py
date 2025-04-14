import sys
import time
import datetime
import json
import numpy as np
import qcodes as qc
from drivers.K6430 import Keithley_6430

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QAction,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from serial.tools import list_ports
import io
import contextlib
from matplotlib import cm


# --- Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


# --- Main Application Window ---
class MeasurementApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keithley 6430 Measurement Application")
        self.resize(1200, 700)

        # Instrument reference; not connected until user clicks Connect.
        self.smu = None

        # Default settings
        self.experiment_name = "Experiment"
        self.nplc_value = 1.0

        # Output folder and auto output flag
        self.output_folder = ""
        self.auto_output = False

        # Flag for back-to-back run
        self.back_to_back = False

        # Live CSV file handles
        self.jv_live_file = None
        self.jt_live_file = None

        # Create Menu Bar and actions
        self.create_menus()

        # Main layout: left panel for controls, right panel for plots.
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Connection Group Box
        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout()
        self.combobox = QComboBox()
        self.refresh_ports()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_instrument)
        conn_layout.addWidget(self.combobox)
        conn_layout.addWidget(self.connect_btn)
        conn_group.setLayout(conn_layout)
        left_layout.addWidget(conn_group)

        # Tabs for measurement controls and settings
        self.control_tabs = QTabWidget()
        left_layout.addWidget(self.control_tabs)

        self.create_jv_tab()
        self.create_jt_tab()
        self.create_settings_tab()

        main_layout.addWidget(left_panel, 0)

        # --- Right Panel (Plot Area) ---
        self.plot_tabs = QTabWidget()
        # JV Plot tab
        self.jv_plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        jv_plot_widget = QWidget()
        jv_plot_layout = QVBoxLayout(jv_plot_widget)
        jv_plot_layout.addWidget(self.jv_plot_canvas)
        self.plot_tabs.addTab(jv_plot_widget, f"JV Plot - {self.experiment_name}")
        # JT Plot tab
        self.jt_plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        jt_plot_widget = QWidget()
        jt_plot_layout = QVBoxLayout(jt_plot_widget)
        jt_plot_layout.addWidget(self.jt_plot_canvas)
        self.plot_tabs.addTab(jt_plot_widget, f"JT Plot - {self.experiment_name}")
        main_layout.addWidget(self.plot_tabs, 1)

        # Timers for measurement loops
        self.jv_timer = QtCore.QTimer()
        self.jv_timer.timeout.connect(self.update_jv_measurement)
        self.jt_timer = QtCore.QTimer()
        self.jt_timer.timeout.connect(self.update_jt_measurement)

        # Data holders for JV sweep
        self.jv_setpoints = None
        self.jv_index = 0
        self.jv_total_cycles = 1
        self.jv_current_cycle = 0
        self.jv_current_cycle_data_voltage = []
        self.jv_current_cycle_data_current = []
        self.jv_cycle_data = []  # list of (voltage, current) arrays per cycle
        self.jv_timer_delay = 0

        # Data holders for JT measurement
        self.jt_start_time = None
        self.jt_time_data = []
        self.jt_current_data = []

    # --- Helper: Live CSV Filename ---
    def get_live_csv_filename(self, measurement_type):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.output_folder if self.output_folder else "."
        return (
            f"{folder}/{self.experiment_name}_{measurement_type}_live_{timestamp}.csv"
        )

    # --- Menu Creation ---
    def create_menus(self):
        menubar = self.menuBar()

        # File Menu: Save/Load Config, Output Folder, Auto Output
        file_menu = menubar.addMenu("File")

        save_config_action = QAction("Save Config", self)
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)

        load_config_action = QAction("Load Config", self)
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)

        select_output_action = QAction("Select Output Folder", self)
        select_output_action.triggered.connect(self.select_output_folder)
        file_menu.addAction(select_output_action)

        self.auto_output_action = QAction("Automatic Output", self, checkable=True)
        self.auto_output_action.triggered.connect(
            lambda checked: self.toggle_auto_output(checked)
        )
        file_menu.addAction(self.auto_output_action)

        # Run Menu: Back-to-Back Run
        run_menu = menubar.addMenu("Run")
        back_to_back_action = QAction("Run JV & JT Back-to-Back", self)
        back_to_back_action.triggered.connect(self.run_back_to_back)
        run_menu.addAction(back_to_back_action)

        # Export Menu: JV and JT exports
        export_menu = menubar.addMenu("Export")
        export_jv_csv_action = QAction("Export JV CSV", self)
        export_jv_csv_action.triggered.connect(self.export_jv_csv)
        export_menu.addAction(export_jv_csv_action)

        export_jv_plot_action = QAction("Export JV Plot", self)
        export_jv_plot_action.triggered.connect(self.export_jv_plot)
        export_menu.addAction(export_jv_plot_action)

        export_jt_csv_action = QAction("Export JT CSV", self)
        export_jt_csv_action.triggered.connect(self.export_jt_csv)
        export_menu.addAction(export_jt_csv_action)

        export_jt_plot_action = QAction("Export JT Plot", self)
        export_jt_plot_action.triggered.connect(self.export_jt_plot)
        export_menu.addAction(export_jt_plot_action)

        # Appearance Menu: Dark Mode toggle
        appearance_menu = menubar.addMenu("Appearance")
        self.dark_mode_action = QAction("Dark Mode", self, checkable=True)
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        appearance_menu.addAction(self.dark_mode_action)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            self.output_folder = folder

    def toggle_auto_output(self, checked):
        self.auto_output = checked

    def toggle_dark_mode(self, checked):
        if checked:
            dark_palette = QtGui.QPalette()
            dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
            dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            dark_palette.setColor(
                QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53)
            )
            dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
            dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
            dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            QApplication.instance().setPalette(dark_palette)
        else:
            QApplication.instance().setPalette(self.style().standardPalette())

    # --- Connection Methods ---
    def refresh_ports(self):
        self.combobox.clear()
        ports = list_ports.comports()
        for port in ports:
            self.combobox.addItem(port.device)

    def connect_instrument(self):
        port = self.combobox.currentText()
        if not port:
            QMessageBox.warning(self, "Connection Error", "No COM port selected.")
            return
        # Convert "COM3" into "ASRL3::INSTR"
        if port.upper().startswith("COM"):
            port_number = port[3:]
            formatted_port = f"ASRL{port_number}::INSTR"
        else:
            formatted_port = port
        try:
            qc.Instrument.close_all()
            self.smu = Keithley_6430("SMU", formatted_port, terminator="\r")
            self.smu.sense_mode("CURR:DC,VOLT:DC,RES")
            self.smu.source_mode("VOLT")
            self.smu.nplc(self.nplc_value)
            QMessageBox.information(
                self, "Connected", f"Connected to instrument on {formatted_port}."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Connection Error", f"Could not connect to instrument:\n{e}"
            )

    # --- JV Sweep Tab ---
    def create_jv_tab(self):
        self.jv_tab = QWidget()
        layout = QVBoxLayout(self.jv_tab)
        param_group = QGroupBox("JV Sweep Parameters")
        form = QFormLayout()
        self.jv_start_voltage = QLineEdit("-10")
        self.jv_stop_voltage = QLineEdit("10")
        # Replace "Number of Points" with "Step Size (V)"
        self.jv_step_size = QLineEdit("1")
        self.jv_speed = QLineEdit("1")  # in V/s
        form.addRow("Start Voltage (V):", self.jv_start_voltage)
        form.addRow("Stop Voltage (V):", self.jv_stop_voltage)
        form.addRow("Step Size (V):", self.jv_step_size)
        form.addRow("Sweep Speed (V/s):", self.jv_speed)
        param_group.setLayout(form)
        layout.addWidget(param_group)

        # Sweep Mode Combobox
        mode_group = QGroupBox("Sweep Mode")
        mode_layout = QHBoxLayout()
        self.jv_mode_combobox = QComboBox()
        self.jv_mode_combobox.addItems(["Standard sweep", "Zero centered"])
        mode_layout.addWidget(self.jv_mode_combobox)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Multiple Cycles Toggle
        cycle_group = QGroupBox("Multiple Cycles")
        cycle_layout = QHBoxLayout()
        self.multiple_cycles_checkbox = QtWidgets.QCheckBox("Enable")
        self.multiple_cycles_checkbox.stateChanged.connect(self.toggle_cycle_input)
        self.jv_cycle_input = QLineEdit("1")
        self.jv_cycle_input.setEnabled(False)
        cycle_layout.addWidget(self.multiple_cycles_checkbox)
        cycle_layout.addWidget(QtWidgets.QLabel("Number of cycles:"))
        cycle_layout.addWidget(self.jv_cycle_input)
        cycle_group.setLayout(cycle_layout)
        layout.addWidget(cycle_group)

        # Start and Stop buttons for JV measurement
        btn_layout = QHBoxLayout()
        self.jv_start_btn = QPushButton("Start JV")
        self.jv_start_btn.clicked.connect(self.start_jv_sweep)
        btn_layout.addWidget(self.jv_start_btn)
        self.jv_stop_btn = QPushButton("Stop JV")
        self.jv_stop_btn.clicked.connect(self.stop_jv_measurement)
        btn_layout.addWidget(self.jv_stop_btn)
        layout.addLayout(btn_layout)

        self.control_tabs.addTab(self.jv_tab, "JV Sweep")

    def toggle_cycle_input(self, state):
        self.jv_cycle_input.setEnabled(state == QtCore.Qt.Checked)

    def start_jv_sweep(self):
        if self.smu is None:
            QMessageBox.warning(
                self, "Instrument Error", "Please connect to the instrument first."
            )
            return
        try:
            start_v = float(self.jv_start_voltage.text())
            stop_v = float(self.jv_stop_voltage.text())
            step_size = float(self.jv_step_size.text())
            sweep_speed = float(self.jv_speed.text())
            if sweep_speed <= 0 or step_size <= 0:
                raise ValueError("Step size and sweep speed must be positive.")
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", f"Invalid JV parameter: {e}")
            return

        # Determine sweep mode and compute setpoints using step size
        mode = self.jv_mode_combobox.currentText()
        if mode == "Standard sweep":
            if start_v < stop_v:
                n_points = int(np.ceil((stop_v - start_v) / step_size)) + 1
            else:
                n_points = int(np.ceil((start_v - stop_v) / step_size)) + 1
            self.jv_setpoints = np.linspace(start_v, stop_v, n_points)
        elif mode == "Zero centered":

            def seg_points(a, b):
                n = int(np.ceil(abs(b - a) / step_size)) + 1
                return np.linspace(a, b, n)

            seg1 = seg_points(0, start_v)
            seg2 = seg_points(start_v, stop_v)[1:]
            seg3 = seg_points(stop_v, 0)[1:]
            self.jv_setpoints = np.concatenate((seg1, seg2, seg3))
        else:
            self.jv_setpoints = np.linspace(start_v, stop_v, 2)

        # Compute delay between steps (time per step = step_size / sweep_speed)
        self.jv_timer_delay = step_size / sweep_speed

        # Handle multiple cycles
        if self.multiple_cycles_checkbox.isChecked():
            try:
                self.jv_total_cycles = int(self.jv_cycle_input.text())
                if self.jv_total_cycles < 1:
                    raise ValueError("Number of cycles must be at least 1.")
            except ValueError as e:
                QMessageBox.warning(
                    self, "Parameter Error", f"Invalid cycle number: {e}"
                )
                return
        else:
            self.jv_total_cycles = 1

        # Reset JV data
        self.jv_index = 0
        self.jv_current_cycle = 0
        self.jv_cycle_data = []
        self.jv_current_cycle_data_voltage = []
        self.jv_current_cycle_data_current = []

        # Open a live CSV file for JV data
        live_filename = self.get_live_csv_filename("JV")
        try:
            self.jv_live_file = open(live_filename, "w")
            if self.multiple_cycles_checkbox.isChecked():
                self.jv_live_file.write("Cycle,Voltage (V),Current (A)\n")
            else:
                self.jv_live_file.write("Voltage (V),Current (A)\n")
        except Exception as e:
            QMessageBox.warning(
                self, "File Error", f"Could not open live CSV file:\n{e}"
            )
            self.jv_live_file = None

        # Set instrument parameters
        self.smu.source_voltage_range(max(abs(start_v), abs(stop_v)))
        self.smu.source_current_compliance(1e-3)
        self.smu.output_enabled(True)

        # Switch to JV plot tab and update title
        self.plot_tabs.setCurrentIndex(0)
        self.jv_plot_canvas.ax.clear()
        self.jv_plot_canvas.ax.set_xlabel("Voltage (V)")
        self.jv_plot_canvas.ax.set_ylabel("Current (A)")
        self.jv_plot_canvas.ax.set_title(f"JV Curve - {self.experiment_name}")
        self.jv_plot_canvas.draw()

        self.jv_timer.start(int(self.jv_timer_delay * 1000))

    def update_jv_measurement(self):
        if self.jv_index < len(self.jv_setpoints):
            voltage_set = self.jv_setpoints[self.jv_index]
            self.smu.source_voltage(voltage_set)
            time.sleep(0.05)
            meas_voltage = self.smu.sense_voltage()
            meas_current = self.smu.sense_current()
            self.jv_current_cycle_data_voltage.append(meas_voltage)
            self.jv_current_cycle_data_current.append(meas_current)
            # Append live CSV row
            if self.jv_live_file:
                if self.multiple_cycles_checkbox.isChecked():
                    self.jv_live_file.write(
                        f"{self.jv_current_cycle + 1},{meas_voltage},{meas_current}\n"
                    )
                else:
                    self.jv_live_file.write(f"{meas_voltage},{meas_current}\n")
                self.jv_live_file.flush()
            self.plot_jv_cycles()
            self.jv_index += 1
        else:
            # End of current cycle
            if self.multiple_cycles_checkbox.isChecked():
                self.jv_cycle_data.append(
                    (
                        np.array(self.jv_current_cycle_data_voltage),
                        np.array(self.jv_current_cycle_data_current),
                    )
                )
            else:
                self.jv_cycle_data = []
            self.jv_current_cycle += 1
            if self.jv_current_cycle < self.jv_total_cycles:
                self.jv_index = 0
                self.jv_current_cycle_data_voltage = []
                self.jv_current_cycle_data_current = []
            else:
                self.jv_timer.stop()
                self.smu.output_enabled(False)
                if self.jv_live_file:
                    self.jv_live_file.close()
                    self.jv_live_file = None
                if self.auto_output and self.output_folder:
                    self.auto_export_jv()
                if self.back_to_back:
                    self.back_to_back = False
                    self.start_jt_measurement()

    def plot_jv_cycles(self):
        self.jv_plot_canvas.ax.clear()
        if self.multiple_cycles_checkbox.isChecked():
            colors = cm.get_cmap("viridis", self.jv_total_cycles)
            # Plot completed cycles
            for idx, (voltages, currents) in enumerate(self.jv_cycle_data):
                self.jv_plot_canvas.ax.plot(
                    voltages,
                    currents,
                    marker="o",
                    color=colors(idx),
                    label=f"Cycle {idx + 1}",
                )
            # Plot current cycle in progress
            if self.jv_current_cycle_data_voltage:
                self.jv_plot_canvas.ax.plot(
                    self.jv_current_cycle_data_voltage,
                    self.jv_current_cycle_data_current,
                    marker="o",
                    color=colors(self.jv_current_cycle),
                    label=f"Cycle {self.jv_current_cycle + 1}",
                )
            self.jv_plot_canvas.ax.legend()
        else:
            self.jv_plot_canvas.ax.plot(
                self.jv_current_cycle_data_voltage,
                self.jv_current_cycle_data_current,
                marker="o",
                color="b",
            )
        self.jv_plot_canvas.ax.set_xlabel("Voltage (V)")
        self.jv_plot_canvas.ax.set_ylabel("Current (A)")
        self.jv_plot_canvas.ax.set_title(f"JV Curve - {self.experiment_name}")
        self.jv_plot_canvas.draw()

    def stop_jv_measurement(self):
        self.jv_timer.stop()
        if self.smu:
            self.smu.output_enabled(False)
        if self.jv_live_file:
            self.jv_live_file.close()
            self.jv_live_file = None

    def export_jv_csv(self):
        if (not self.jv_cycle_data) and (not self.jv_current_cycle_data_voltage):
            QMessageBox.warning(self, "Export Error", "No JV data available.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save JV Data",
            f"{self.experiment_name}_JV_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)",
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(f"Experiment: {self.experiment_name}\n")
                    if self.multiple_cycles_checkbox.isChecked():
                        f.write(f"Number of cycles: {self.jv_total_cycles}\n")
                        f.write("Cycle,Voltage (V),Current (A)\n")
                        cycle_num = 1
                        for voltages, currents in self.jv_cycle_data:
                            for v, i in zip(voltages, currents):
                                f.write(f"{cycle_num},{v},{i}\n")
                            cycle_num += 1
                        if self.jv_current_cycle_data_voltage:
                            for v, i in zip(
                                self.jv_current_cycle_data_voltage,
                                self.jv_current_cycle_data_current,
                            ):
                                f.write(f"{cycle_num},{v},{i}\n")
                    else:
                        f.write("Voltage (V),Current (A)\n")
                        for v, i in zip(
                            self.jv_current_cycle_data_voltage,
                            self.jv_current_cycle_data_current,
                        ):
                            f.write(f"{v},{i}\n")
                QMessageBox.information(self, "Export", "JV CSV exported successfully.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export CSV:\n{e}"
                )

    def export_jv_plot(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save JV Plot",
            f"{self.experiment_name}_JV_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png)",
        )
        if filename:
            try:
                self.jv_plot_canvas.fig.savefig(filename)
                QMessageBox.information(
                    self, "Export", "JV plot exported successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export plot:\n{e}"
                )

    def auto_export_jv(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.output_folder}/{self.experiment_name}_JV_{timestamp}.csv"
        plot_filename = (
            f"{self.output_folder}/{self.experiment_name}_JV_{timestamp}.png"
        )
        try:
            with open(csv_filename, "w") as f:
                f.write(f"Experiment: {self.experiment_name}\n")
                if self.multiple_cycles_checkbox.isChecked():
                    f.write(f"Number of cycles: {self.jv_total_cycles}\n")
                    f.write("Cycle,Voltage (V),Current (A)\n")
                    cycle_num = 1
                    for voltages, currents in self.jv_cycle_data:
                        for v, i in zip(voltages, currents):
                            f.write(f"{cycle_num},{v},{i}\n")
                        cycle_num += 1
                    if self.jv_current_cycle_data_voltage:
                        for v, i in zip(
                            self.jv_current_cycle_data_voltage,
                            self.jv_current_cycle_data_current,
                        ):
                            f.write(f"{cycle_num},{v},{i}\n")
                else:
                    f.write("Voltage (V),Current (A)\n")
                    for v, i in zip(
                        self.jv_current_cycle_data_voltage,
                        self.jv_current_cycle_data_current,
                    ):
                        f.write(f"{v},{i}\n")
            self.jv_plot_canvas.fig.savefig(plot_filename)
        except Exception as e:
            QMessageBox.critical(
                self, "Auto Export Error", f"Failed to auto-export JV data:\n{e}"
            )

    # --- JT Tab ---
    def create_jt_tab(self):
        self.jt_tab = QWidget()
        layout = QVBoxLayout(self.jt_tab)
        param_group = QGroupBox("JT Parameters")
        form = QFormLayout()
        self.jt_hold_voltage = QLineEdit("5")
        # Create a composite widget for Total Time with units
        total_time_widget = QWidget()
        total_time_layout = QHBoxLayout(total_time_widget)
        total_time_layout.setContentsMargins(0, 0, 0, 0)
        self.jt_total_time = QLineEdit("30")
        self.jt_total_time_unit = QComboBox()
        self.jt_total_time_unit.addItems(["seconds", "minutes", "hours"])
        total_time_layout.addWidget(self.jt_total_time)
        total_time_layout.addWidget(self.jt_total_time_unit)
        # Create a composite widget for Sample Interval with units
        sample_interval_widget = QWidget()
        sample_interval_layout = QHBoxLayout(sample_interval_widget)
        sample_interval_layout.setContentsMargins(0, 0, 0, 0)
        self.jt_sample_interval = QLineEdit("0.5")
        self.jt_sample_interval_unit = QComboBox()
        self.jt_sample_interval_unit.addItems(["seconds", "minutes", "hours"])
        sample_interval_layout.addWidget(self.jt_sample_interval)
        sample_interval_layout.addWidget(self.jt_sample_interval_unit)
        form.addRow("Hold Voltage (V):", self.jt_hold_voltage)
        form.addRow("Total Time:", total_time_widget)
        form.addRow("Sample Interval:", sample_interval_widget)
        param_group.setLayout(form)
        layout.addWidget(param_group)
        # Start and Stop buttons for JT measurement
        btn_layout = QHBoxLayout()
        self.jt_start_btn = QPushButton("Start JT")
        self.jt_start_btn.clicked.connect(self.start_jt_measurement)
        btn_layout.addWidget(self.jt_start_btn)
        self.jt_stop_btn = QPushButton("Stop JT")
        self.jt_stop_btn.clicked.connect(self.stop_jt_measurement)
        btn_layout.addWidget(self.jt_stop_btn)
        layout.addLayout(btn_layout)
        self.control_tabs.addTab(self.jt_tab, "JT")

    def start_jt_measurement(self):
        if self.smu is None:
            QMessageBox.warning(
                self, "Instrument Error", "Please connect to the instrument first."
            )
            return
        try:
            hold_voltage = float(self.jt_hold_voltage.text())
            total_time_val = float(self.jt_total_time.text())
            sample_interval_val = float(self.jt_sample_interval.text())
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error", f"Invalid JT parameter: {e}")
            return
        # Convert time values to seconds based on selected units
        units = {"seconds": 1, "minutes": 60, "hours": 3600}
        total_time_factor = units[self.jt_total_time_unit.currentText()]
        sample_interval_factor = units[self.jt_sample_interval_unit.currentText()]
        total_time_seconds = total_time_val * total_time_factor
        sample_interval_seconds = sample_interval_val * sample_interval_factor
        if total_time_seconds <= 0 or sample_interval_seconds <= 0:
            QMessageBox.warning(
                self, "Parameter Error", "Time values must be positive."
            )
            return

        self.smu.source_voltage(hold_voltage)
        self.smu.output_enabled(True)
        self.jt_start_time = time.time()
        self.jt_time_data = []
        self.jt_current_data = []

        # Open live CSV file for JT data
        live_filename = self.get_live_csv_filename("JT")
        try:
            self.jt_live_file = open(live_filename, "w")
            self.jt_live_file.write("Time (s),Current (A)\n")
        except Exception as e:
            QMessageBox.warning(
                self, "File Error", f"Could not open live CSV file for JT:\n{e}"
            )
            self.jt_live_file = None

        self.plot_tabs.setCurrentIndex(1)
        self.jt_plot_canvas.ax.clear()
        self.jt_plot_canvas.ax.set_xlabel("Time (s)")
        self.jt_plot_canvas.ax.set_ylabel("Current (A)")
        self.jt_plot_canvas.ax.set_title(f"JT Curve - {self.experiment_name}")
        self.jt_plot_canvas.draw()
        self.jt_timer.start(int(sample_interval_seconds * 1000))
        # Store total time in seconds for checking in the update
        self.jt_total_time_seconds = total_time_seconds

    def update_jt_measurement(self):
        elapsed = time.time() - self.jt_start_time
        if elapsed > self.jt_total_time_seconds:
            self.stop_jt_measurement()
            return
        current = self.smu.sense_current()
        self.jt_time_data.append(elapsed)
        self.jt_current_data.append(current)
        # Append live CSV row
        if self.jt_live_file:
            self.jt_live_file.write(f"{elapsed},{current}\n")
            self.jt_live_file.flush()
        self.jt_plot_canvas.ax.clear()
        self.jt_plot_canvas.ax.plot(self.jt_time_data, self.jt_current_data, marker="o")
        self.jt_plot_canvas.ax.set_xlabel("Time (s)")
        self.jt_plot_canvas.ax.set_ylabel("Current (A)")
        self.jt_plot_canvas.ax.set_title(f"JT Curve - {self.experiment_name}")
        self.jt_plot_canvas.draw()

    def stop_jt_measurement(self):
        self.jt_timer.stop()
        if self.smu:
            self.smu.output_enabled(False)
        if self.jt_live_file:
            self.jt_live_file.close()
            self.jt_live_file = None
        if self.auto_output and self.output_folder:
            self.auto_export_jt()

    def export_jt_csv(self):
        if not self.jt_time_data or not self.jt_current_data:
            QMessageBox.warning(self, "Export Error", "No JT data available.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save JT Data",
            f"{self.experiment_name}_JT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)",
        )
        if filename:
            try:
                data = np.column_stack((self.jt_time_data, self.jt_current_data))
                header = "Time (s),Current (A)"
                np.savetxt(filename, data, delimiter=",", header=header, comments="")
                QMessageBox.information(self, "Export", "JT CSV exported successfully.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export CSV:\n{e}"
                )

    def export_jt_plot(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save JT Plot",
            f"{self.experiment_name}_JT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png)",
        )
        if filename:
            try:
                self.jt_plot_canvas.fig.savefig(filename)
                QMessageBox.information(
                    self, "Export", "JT plot exported successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export plot:\n{e}"
                )

    def auto_export_jt(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.output_folder}/{self.experiment_name}_JT_{timestamp}.csv"
        plot_filename = (
            f"{self.output_folder}/{self.experiment_name}_JT_{timestamp}.png"
        )
        try:
            data = np.column_stack((self.jt_time_data, self.jt_current_data))
            header = "Time (s),Current (A)"
            np.savetxt(csv_filename, data, delimiter=",", header=header, comments="")
            self.jt_plot_canvas.fig.savefig(plot_filename)
        except Exception as e:
            QMessageBox.critical(
                self, "Auto Export Error", f"Failed to auto-export JT data:\n{e}"
            )

    # --- Settings Tab ---
    def create_settings_tab(self):
        self.settings_tab = QWidget()
        layout = QVBoxLayout(self.settings_tab)
        exp_group = QGroupBox("Experiment Settings")
        exp_form = QFormLayout()
        self.exp_name_edit = QLineEdit(self.experiment_name)
        self.nplc_edit = QLineEdit(str(self.nplc_value))
        exp_form.addRow("Experiment Name:", self.exp_name_edit)
        exp_form.addRow("nplc Value:", self.nplc_edit)
        exp_group.setLayout(exp_form)
        layout.addWidget(exp_group)

        # Sourcemeter Utilities
        smu_group = QGroupBox("Sourcemeter Utilities")
        smu_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.dump_btn = QPushButton("Dump Sourcemeter Config Info")
        self.dump_btn.clicked.connect(self.dump_smu_config)
        btn_layout.addWidget(self.dump_btn)
        self.reset_btn = QPushButton("Reset Sourcemeter")
        self.reset_btn.clicked.connect(self.reset_smu)
        btn_layout.addWidget(self.reset_btn)
        smu_layout.addLayout(btn_layout)
        self.config_info_textbox = QPlainTextEdit()
        self.config_info_textbox.setReadOnly(True)
        smu_layout.addWidget(self.config_info_textbox)
        smu_group.setLayout(smu_layout)
        layout.addWidget(smu_group)

        # Apply Settings Button
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_settings_btn)

        self.control_tabs.addTab(self.settings_tab, "Settings")

    def apply_settings(self):
        self.experiment_name = self.exp_name_edit.text().strip() or "Experiment"
        try:
            self.nplc_value = float(self.nplc_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Settings Error", "Invalid nplc value.")
            return
        if self.smu is not None:
            try:
                self.smu.nplc(self.nplc_value)
            except Exception as e:
                QMessageBox.warning(
                    self, "Instrument Error", f"Could not set nplc:\n{e}"
                )
        self.plot_tabs.setTabText(0, f"JV Plot - {self.experiment_name}")
        self.plot_tabs.setTabText(1, f"JT Plot - {self.experiment_name}")
        QMessageBox.information(self, "Settings", "Settings applied successfully.")

    def dump_smu_config(self):
        if self.smu is None:
            QMessageBox.warning(self, "Instrument Error", "Instrument not connected.")
            return
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.smu.print_readable_snapshot(update=True)
            config_text = buf.getvalue()
            self.config_info_textbox.setPlainText(config_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to dump config info:\n{e}")

    def reset_smu(self):
        if self.smu is None:
            QMessageBox.warning(self, "Instrument Error", "Instrument not connected.")
            return
        try:
            self.smu.reset()
            QMessageBox.information(self, "Reset", "Sourcemeter has been reset.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reset sourcemeter:\n{e}")

    def save_config(self):
        config = {
            "experiment_name": self.exp_name_edit.text(),
            "nplc_value": self.nplc_edit.text(),
            "jv_start_voltage": self.jv_start_voltage.text(),
            "jv_stop_voltage": self.jv_stop_voltage.text(),
            "jv_step_size": self.jv_step_size.text(),
            "jv_speed": self.jv_speed.text(),
            "jv_mode": self.jv_mode_combobox.currentText(),
            "multiple_cycles": self.multiple_cycles_checkbox.isChecked(),
            "jv_cycles": self.jv_cycle_input.text(),
            "jt_hold_voltage": self.jt_hold_voltage.text(),
            "jt_total_time": self.jt_total_time.text(),
            "jt_total_time_unit": self.jt_total_time_unit.currentText(),
            "jt_sample_interval": self.jt_sample_interval.text(),
            "jt_sample_interval_unit": self.jt_sample_interval_unit.currentText(),
        }
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "config.json", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    json.dump(config, f, indent=4)
                QMessageBox.information(
                    self, "Save Config", "Configuration saved successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Config", f"Failed to save configuration:\n{e}"
                )

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "JSON Files (*.json)"
        )
        if filename:
            try:
                with open(filename, "r") as f:
                    config = json.load(f)
                self.exp_name_edit.setText(config.get("experiment_name", "Experiment"))
                self.nplc_edit.setText(str(config.get("nplc_value", "1.0")))
                self.jv_start_voltage.setText(config.get("jv_start_voltage", "-10"))
                self.jv_stop_voltage.setText(config.get("jv_stop_voltage", "10"))
                self.jv_step_size.setText(config.get("jv_step_size", "1"))
                self.jv_speed.setText(config.get("jv_speed", "1"))
                self.jv_mode_combobox.setCurrentText(
                    config.get("jv_mode", "Standard sweep")
                )
                self.multiple_cycles_checkbox.setChecked(
                    config.get("multiple_cycles", False)
                )
                self.jv_cycle_input.setText(config.get("jv_cycles", "1"))
                self.jt_hold_voltage.setText(config.get("jt_hold_voltage", "5"))
                self.jt_total_time.setText(config.get("jt_total_time", "30"))
                self.jt_total_time_unit.setCurrentText(
                    config.get("jt_total_time_unit", "seconds")
                )
                self.jt_sample_interval.setText(config.get("jt_sample_interval", "0.5"))
                self.jt_sample_interval_unit.setCurrentText(
                    config.get("jt_sample_interval_unit", "seconds")
                )
                QMessageBox.information(
                    self, "Load Config", "Configuration loaded successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Load Config", f"Failed to load configuration:\n{e}"
                )

    def run_back_to_back(self):
        if self.smu is None:
            QMessageBox.warning(
                self, "Instrument Error", "Please connect to the instrument first."
            )
            return
        self.back_to_back = True
        self.start_jv_sweep()


def main():
    qc.Config.logging["console"]["format"] = "%(asctime)s - %(message)s"
    app = QApplication(sys.argv)
    win = MeasurementApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
