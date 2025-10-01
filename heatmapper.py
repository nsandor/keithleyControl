#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QLineEdit, QLabel, QSlider,
    QMessageBox, QGroupBox, QFormLayout, QColorDialog, QInputDialog
)
from PyQt5.QtCore import Qt

# ------------------------------ Plot Canvas ------------------------------
class HeatmapCanvas(FigureCanvas):
    """A canvas that updates the plot."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.setParent(parent)
        self.image = None
        self.colorbar = None

    def plot(self, data, cmap='viridis', vmin=None, vmax=None, log_scale=False,
             title="", colorbar_label="", show_values=False,
             bad_pixel_coords=None, bad_pixel_color=None):
        """Render a heatmap with optional overlays."""
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.image = None
        self.colorbar = None

        if data is not None:
            # For log plotting, ensure strictly positive inputs to the norm.
            data_to_plot = np.array(data, copy=True)
            if log_scale:
                # Avoid nonpositive values with a small floor.
                eps = 1e-12
                data_to_plot = np.where(data_to_plot <= 0, eps, data_to_plot)
                self.image = self.axes.imshow(
                    data_to_plot, cmap=cmap,
                    norm=LogNorm(vmin=vmin if vmin else None, vmax=vmax if vmax else None),
                    interpolation='nearest'
                )
            else:
                self.image = self.axes.imshow(
                    data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest'
                )

            self.colorbar = self.fig.colorbar(self.image, ax=self.axes)
            if colorbar_label:
                self.colorbar.set_label(colorbar_label)
            if title:
                self.axes.set_title(title)

            # Overlay bad pixels
            if bad_pixel_coords and bad_pixel_color:
                rgba = to_rgba(bad_pixel_color, alpha=0.85)
                for (r, c) in bad_pixel_coords:
                    rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     facecolor=rgba, edgecolor='black', linewidth=0.5)
                    self.axes.add_patch(rect)

            # Optional numeric labels
            if show_values:
                h, w = data.shape
                for i in range(h):
                    for j in range(w):
                        value = data[i, j]
                        text = f'{value:.2f}' if abs(value) < 1000 else f'{value:.1e}'
                        self.axes.text(j, i, text, ha='center', va='center',
                                       color='white', fontsize=8, weight='bold')

        self.draw()

    def add_annotation(self, x, y, text, color='black'):
        """Adds a text annotation to the plot."""
        if self.image:
            self.axes.text(x, y, text, color=color, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.5, pad=2))
            self.draw()

# ------------------------------ Main Window ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photodiode Data Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Data storage
        self.datasets = {}            # name -> np.array(10x10)
        self.current_data_key = None
        self.subtract_data_key = None
        self.divide_data_key = None
        self.result_data = None
        self.operation = None         # None | 'subtract' | 'divide'

        # Annotation handling
        self.annotation_cid = None

        # Bad pixels
        self.bad_pixel_coords = []    # list of (row,col)
        self.bad_pixel_color = '#FF0000'

        # Globals for batch mode
        self.batch_root = None
        self.global_subtract_array = None
        self.global_divide_array = None

        # Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ------------------------- Left Panel -------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(360)

        # File Operations
        file_group = QGroupBox("File Operations")
        file_layout = QFormLayout()
        self.import_button = QPushButton("Import CSV")
        self.export_button = QPushButton("Export Current View (CSV)")
        self.export_img_button = QPushButton("Export as Image")
        file_layout.addRow(self.import_button)
        file_layout.addRow(self.export_button)
        file_layout.addRow(self.export_img_button)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Data Selection
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout()
        self.data_selector = QComboBox()
        self.subtract_selector = QComboBox()
        self.divide_selector = QComboBox()
        self.subtract_button = QPushButton("Subtract Selected")
        self.divide_button = QPushButton("Divide Selected")
        self.clear_subtract_button = QPushButton("Clear Subtraction")
        self.clear_divide_button = QPushButton("Clear Division")
        data_layout.addRow(QLabel("Primary Data:"), self.data_selector)
        data_layout.addRow(QLabel("Subtract Data:"), self.subtract_selector)
        data_layout.addRow(self.subtract_button)
        data_layout.addRow(self.clear_subtract_button)
        data_layout.addRow(QLabel("Divide Data:"), self.divide_selector)
        data_layout.addRow(self.divide_button)
        data_layout.addRow(self.clear_divide_button)
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        # Plot Appearance
        appearance_group = QGroupBox("Plot Appearance")
        appearance_layout = QFormLayout()
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Enter plot title")
        self.colorbar_label_input = QLineEdit()
        self.colorbar_label_input.setPlaceholderText("Enter colorbar label (e.g., 'Current')")
        self.units_selector = QComboBox()
        self.units_selector.addItems(['pA', 'nA', 'µA', 'mA'])
        self.units_selector.setCurrentText('µA')
        self.show_values_toggle = QPushButton("Show Values: OFF")
        self.show_values_toggle.setCheckable(True)
        appearance_layout.addRow(QLabel("Plot Title:"), self.title_input)
        appearance_layout.addRow(QLabel("Colorbar Label:"), self.colorbar_label_input)
        appearance_layout.addRow(QLabel("Units:"), self.units_selector)
        appearance_layout.addRow(self.show_values_toggle)
        appearance_group.setLayout(appearance_layout)
        left_layout.addWidget(appearance_group)

        # Heatmap Controls
        heatmap_group = QGroupBox("Heatmap Controls")
        heatmap_layout = QFormLayout()
        self.cmap_selector = QComboBox()
        self.cmap_selector.addItems(plt.colormaps())
        self.vmin_input = QLineEdit()
        self.vmin_input.setPlaceholderText("auto")
        self.vmax_input = QLineEdit()
        self.vmax_input.setPlaceholderText("auto")
        self.log_scale_toggle = QPushButton("Toggle Log Scale: OFF")
        self.log_scale_toggle.setCheckable(True)
        self.update_range_button = QPushButton("Update Range")
        heatmap_layout.addRow(QLabel("Colormap:"), self.cmap_selector)
        heatmap_layout.addRow(QLabel("Min Value:"), self.vmin_input)
        heatmap_layout.addRow(QLabel("Max Value:"), self.vmax_input)
        heatmap_layout.addRow(self.update_range_button)
        heatmap_layout.addRow(self.log_scale_toggle)
        heatmap_group.setLayout(heatmap_layout)
        left_layout.addWidget(heatmap_group)

        # Bad Pixels
        bad_group = QGroupBox("Bad Pixels")
        bad_layout = QFormLayout()
        self.bad_indices_input = QLineEdit()
        self.bad_indices_input.setPlaceholderText("e.g. 1,5,12-15,87")
        self.bad_color_button = QPushButton("Select Color")
        self.bad_color_display = QLabel()
        self.bad_color_display.setFixedHeight(22)
        self.bad_color_display.setStyleSheet(
            f"background-color: {self.bad_pixel_color}; border: 1px solid #444;")
        bad_layout.addRow(QLabel("Indices:"), self.bad_indices_input)
        bad_layout.addRow(self.bad_color_button, self.bad_color_display)
        bad_group.setLayout(bad_layout)
        left_layout.addWidget(bad_group)

        # Annotations
        annotation_group = QGroupBox("Annotations")
        annotation_layout = QFormLayout()
        self.add_text_button = QPushButton("Add Text Annotation")
        annotation_layout.addRow(self.add_text_button)
        annotation_group.setLayout(annotation_layout)
        left_layout.addWidget(annotation_group)

        # --------------------- Batch Processing ----------------------
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QFormLayout()

        self.batch_root_display = QLineEdit()
        self.batch_root_display.setReadOnly(True)
        self.choose_root_btn = QPushButton("Choose Root Folder")

        self.global_sub_display = QLineEdit()
        self.global_sub_display.setReadOnly(True)
        self.choose_global_sub_btn = QPushButton("Global Subtract CSV (optional)")

        self.global_div_display = QLineEdit()
        self.global_div_display.setReadOnly(True)
        self.choose_global_div_btn = QPushButton("Global Divide CSV (optional)")

        self.run_batch_btn = QPushButton("Run Batch → processed/")
        self.run_batch_btn.setStyleSheet("font-weight: bold;")

        batch_layout.addRow(QLabel("Root Folder:"), self.batch_root_display)
        batch_layout.addRow(self.choose_root_btn)
        batch_layout.addRow(QLabel("Subtract Ref:"), self.global_sub_display)
        batch_layout.addRow(self.choose_global_sub_btn)
        batch_layout.addRow(QLabel("Divide Ref:"), self.global_div_display)
        batch_layout.addRow(self.choose_global_div_btn)
        batch_layout.addRow(self.run_batch_btn)
        batch_group.setLayout(batch_layout)
        left_layout.addWidget(batch_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ------------------------- Right Panel -------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.canvas = HeatmapCanvas(self, width=8, height=6, dpi=100)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_panel)

        # ------------------ Signals & Slots ------------------
        self.import_button.clicked.connect(self.import_csv)
        self.export_button.clicked.connect(self.export_csv)
        self.export_img_button.clicked.connect(self.export_image)

        self.data_selector.currentIndexChanged.connect(self.select_primary_data)
        self.subtract_selector.currentIndexChanged.connect(self.select_subtract_data)
        self.divide_selector.currentIndexChanged.connect(self.select_divide_data)
        self.subtract_button.clicked.connect(self.perform_subtraction)
        self.divide_button.clicked.connect(self.perform_division)
        self.clear_subtract_button.clicked.connect(self.clear_subtraction)
        self.clear_divide_button.clicked.connect(self.clear_division)

        self.cmap_selector.currentTextChanged.connect(self.update_plot)
        self.update_range_button.clicked.connect(self.update_plot)
        self.log_scale_toggle.clicked.connect(self.toggle_log_scale)
        self.add_text_button.clicked.connect(self.add_annotation)
        self.title_input.textChanged.connect(self.update_plot)
        self.colorbar_label_input.textChanged.connect(self.update_plot)
        self.units_selector.currentTextChanged.connect(self.update_plot)
        self.show_values_toggle.clicked.connect(self.toggle_show_values)

        self.bad_indices_input.textChanged.connect(self.update_bad_pixels)
        self.bad_color_button.clicked.connect(self.choose_bad_pixel_color)

        # Batch actions
        self.choose_root_btn.clicked.connect(self.choose_batch_root)
        self.choose_global_sub_btn.clicked.connect(self.choose_global_subtract_csv)
        self.choose_global_div_btn.clicked.connect(self.choose_global_divide_csv)
        self.run_batch_btn.clicked.connect(self.run_batch_processing)

        self.update_selectors()

    # -------------------------- Helpers --------------------------
    def choose_bad_pixel_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bad_pixel_color = color.name()
            self.bad_color_display.setStyleSheet(
                f"background-color: {self.bad_pixel_color}; border: 1px solid #444;")
            self.update_plot()

    def update_bad_pixels(self):
        text = self.bad_indices_input.text().strip()
        coords = []
        if text:
            try:
                indices = self._parse_indices(text)
                for idx in indices:
                    if 1 <= idx <= 100:  # 1-based indexing
                        r = (idx - 1) // 10
                        c = (idx - 1) % 10
                        coords.append((r, c))
            except ValueError:
                pass
        self.bad_pixel_coords = coords
        self.update_plot()

    def _parse_indices(self, s):
        """Parse a string like '1,2,5-8,20' into a list of ints (1-based)."""
        result = []
        tokens = [t.strip() for t in s.split(',') if t.strip()]
        for tok in tokens:
            if '-' in tok:
                parts = tok.split('-')
                if len(parts) != 2:
                    raise ValueError(f"Invalid range token: {tok}")
                start_i = int(parts[0]); end_i = int(parts[1])
                if start_i > end_i:
                    start_i, end_i = end_i, start_i
                result.extend(range(start_i, end_i + 1))
            else:
                result.append(int(tok))
        return result

    # -------------------------- File I/O --------------------------
    def import_csv(self):
        """Import a single 10x10 CSV file into datasets."""
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                data = np.loadtxt(path, delimiter=',')
                if data.shape != (10, 10):
                    raise ValueError("Data is not 10x10.")
                filename = os.path.basename(path)
                self.datasets[filename] = data
                self.update_selectors()
                self.data_selector.setCurrentText(filename)
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import file: {e}")

    def export_csv(self):
        """Export the currently displayed data to a CSV file."""
        if self.result_data is not None:
            data_to_save = self.result_data
            op_name = 'division' if self.operation == 'divide' else 'subtraction'
            default_name = f"{op_name}_result.csv"
        elif self.current_data_key:
            data_to_save = self.datasets[self.current_data_key]
            default_name = f"exported_{self.current_data_key}"
        else:
            QMessageBox.warning(self, "Export Error", "No data to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", default_name, "CSV Files (*.csv)")
        if path:
            try:
                np.savetxt(path, data_to_save, delimiter=',')
                QMessageBox.information(self, "Success", f"Data exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export file: {e}")

    def export_image(self):
        """Export the current heatmap view as a raster image."""
        if self.current_data_key is None and self.result_data is None:
            QMessageBox.warning(self, "Export Error", "No plot to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Image", "heatmap.png",
                                              "PNG Files (*.png);;JPEG Files (*.jpg)")
        if path:
            try:
                self.canvas.fig.savefig(path, dpi=300)
                QMessageBox.information(self, "Success", f"Image saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save image: {e}")

    def update_selectors(self):
        """Update the data selection dropdowns."""
        keys = list(self.datasets.keys())
        self.data_selector.clear()
        self.data_selector.addItems(keys)
        self.subtract_selector.clear()
        self.subtract_selector.addItem("None")
        self.subtract_selector.addItems(keys)
        self.divide_selector.clear()
        self.divide_selector.addItem("None")
        self.divide_selector.addItems(keys)

    # -------------------------- Data Selection --------------------------
    def select_primary_data(self, index):
        if index >= 0:
            self.current_data_key = self.data_selector.itemText(index)
            self.result_data = None
            self.operation = None
            self.update_plot()

    def select_subtract_data(self, index):
        self.subtract_data_key = self.subtract_selector.itemText(index) if index > 0 else None

    def select_divide_data(self, index):
        self.divide_data_key = self.divide_selector.itemText(index) if index > 0 else None

    # -------------------------- Operations --------------------------
    def perform_subtraction(self):
        """Subtract the chosen dataset from the primary dataset."""
        if self.current_data_key and self.subtract_data_key:
            a = self.datasets[self.current_data_key]
            b = self.datasets[self.subtract_data_key]
            if a.shape != b.shape:
                QMessageBox.warning(self, "Subtraction Error", "Datasets must have the same shape.")
                return
            result = a - b
            result[result == 0] = 1e-10  # log-safe
            self.result_data = result
            self.operation = 'subtract'
            self.update_plot()
        else:
            QMessageBox.warning(self, "Subtraction Error",
                                "Please select both a primary and a subtract dataset.")

    def perform_division(self):
        """Divide the primary dataset by the chosen dataset."""
        if self.current_data_key and self.divide_data_key:
            a = self.datasets[self.current_data_key]
            b = self.datasets[self.divide_data_key]
            if a.shape != b.shape:
                QMessageBox.warning(self, "Division Error", "Datasets must have the same shape.")
                return
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.true_divide(a, b)
                result[~np.isfinite(result)] = 0.0
            result[result == 0] = 1e-10
            self.result_data = result
            self.operation = 'divide'
            self.update_plot()
        else:
            QMessageBox.warning(self, "Division Error",
                                "Please select both a primary and a divide dataset.")

    def clear_subtraction(self):
        self.result_data = None
        self.operation = None
        self.subtract_selector.setCurrentIndex(0)
        self.update_plot()

    def clear_division(self):
        self.result_data = None
        self.operation = None
        self.divide_selector.setCurrentIndex(0)
        self.update_plot()

    # -------------------------- Plot Controls --------------------------
    def toggle_log_scale(self):
        if self.log_scale_toggle.isChecked():
            self.log_scale_toggle.setText("Toggle Log Scale: ON")
        else:
            self.log_scale_toggle.setText("Toggle Log Scale: OFF")
        self.update_plot()

    def get_unit_conversion_factor(self, target_unit):
        return {
            'pA': 1e12,
            'nA': 1e9,
            'µA': 1e6,
            'mA': 1e3
        }.get(target_unit, 1)

    def convert_data_units(self, data, target_unit):
        return data * self.get_unit_conversion_factor(target_unit)

    def get_colorbar_label(self):
        base_label = self.colorbar_label_input.text().strip()
        units = self.units_selector.currentText()
        if self.operation == 'divide':
            return f"{base_label} (ratio)" if base_label else "Ratio"
        return f"{base_label} ({units})" if base_label else f"Current ({units})"

    def toggle_show_values(self):
        self.show_values_toggle.setText(
            "Show Values: ON" if self.show_values_toggle.isChecked() else "Show Values: OFF"
        )
        self.update_plot()

    def add_annotation(self):
        if self.current_data_key is None and self.result_data is None:
            QMessageBox.warning(self, "Annotation Error", "Please load data first.")
            return
        text, ok = QInputDialog.getText(self, 'Add Annotation', 'Enter text:')
        if ok and text:
            color = QColorDialog.getColor()
            if color.isValid():
                QMessageBox.information(self, "Place Annotation",
                                        "Click on the plot to place the text.")
                if self.annotation_cid:
                    self.canvas.figure.canvas.mpl_disconnect(self.annotation_cid)
                self.annotation_cid = self.canvas.figure.canvas.mpl_connect(
                    'button_press_event',
                    lambda event: self._place_annotation(event, text, color.name())
                )

    def _place_annotation(self, event, text, color_name):
        if event.inaxes == self.canvas.axes:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.canvas.add_annotation(x, y, text, color=color_name)
            if self.annotation_cid:
                self.canvas.figure.canvas.mpl_disconnect(self.annotation_cid)
                self.annotation_cid = None

    def update_plot(self):
        data_to_plot = self.result_data if self.result_data is not None else (
            self.datasets.get(self.current_data_key) if self.current_data_key else None
        )
        if data_to_plot is None:
            self.canvas.plot(None)
            return

        # Units (skip for ratio)
        if self.operation == 'divide':
            converted = data_to_plot
        else:
            converted = self.convert_data_units(data_to_plot, self.units_selector.currentText())

        cmap = self.cmap_selector.currentText()
        try:
            vmin = float(self.vmin_input.text()) if self.vmin_input.text() else None
            vmax = float(self.vmax_input.text()) if self.vmax_input.text() else None
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid range. Enter numbers only.")
            vmin, vmax = None, None

        log_scale = self.log_scale_toggle.isChecked()
        title = self.title_input.text()
        cblabel = self.get_colorbar_label()
        show_values = self.show_values_toggle.isChecked()

        self.canvas.plot(
            converted, cmap=cmap, vmin=vmin, vmax=vmax, log_scale=log_scale,
            title=title, colorbar_label=cblabel, show_values=show_values,
            bad_pixel_coords=self.bad_pixel_coords, bad_pixel_color=self.bad_pixel_color
        )

    # -------------------------- Batch UI --------------------------
    def choose_batch_root(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Root Folder", "")
        if path:
            self.batch_root = path
            self.batch_root_display.setText(path)

    def _load_csv_checked(self, path, label):
        try:
            arr = np.loadtxt(path, delimiter=',')
        except Exception as e:
            raise ValueError(f"{label}: failed to load '{path}': {e}")
        if arr.shape != (10, 10):
            raise ValueError(f"{label}: '{path}' is not 10x10.")
        return arr

    def choose_global_subtract_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Global Subtract CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.global_subtract_array = self._load_csv_checked(path, "Subtract Ref")
                self.global_sub_display.setText(path)
                # Set operation if not already chosen
                if self.operation is None:
                    self.operation = 'subtract'
            except Exception as e:
                self.global_subtract_array = None
                self.global_sub_display.clear()
                QMessageBox.critical(self, "Load Error", str(e))

    def choose_global_divide_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Global Divide CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.global_divide_array = self._load_csv_checked(path, "Divide Ref")
                self.global_div_display.setText(path)
                if self.operation is None:
                    self.operation = 'divide'
            except Exception as e:
                self.global_divide_array = None
                self.global_div_display.clear()
                QMessageBox.critical(self, "Load Error", str(e))

    # -------------------------- Batch Core --------------------------
    def run_batch_processing(self):
        """
        Process every subfolder of self.batch_root:
          - find exactly one CSV in each subfolder,
          - apply selected operation (subtract/divide) using global reference if provided,
          - render heatmap with current GUI settings,
          - save to <root>/processed/<subfolder>.png
        """
        if not self.batch_root or not os.path.isdir(self.batch_root):
            QMessageBox.warning(self, "Batch Error", "Choose a valid Root Folder.")
            return

        # Decide operation & references
        op = self.operation
        sub_ref = self.global_subtract_array
        div_ref = self.global_divide_array

        if op == 'subtract' and sub_ref is None and self.subtract_data_key:
            sub_ref = self.datasets.get(self.subtract_data_key)
        if op == 'divide' and div_ref is None and self.divide_data_key:
            div_ref = self.datasets.get(self.divide_data_key)

        # Output folder
        out_dir = os.path.join(self.batch_root, "processed")
        os.makedirs(out_dir, exist_ok=True)

        # Gather capture subfolders
        subfolders = [d for d in sorted(os.listdir(self.batch_root))
                      if os.path.isdir(os.path.join(self.batch_root, d))]
        if not subfolders:
            QMessageBox.warning(self, "Batch Error", "No subfolders found in Root Folder.")
            return

        failures = []
        processed = 0

        # Cache GUI plot settings once
        cmap = self.cmap_selector.currentText()
        log_scale = self.log_scale_toggle.isChecked()
        title_base = self.title_input.text().strip()
        cblabel = self.get_colorbar_label()
        show_values = self.show_values_toggle.isChecked()
        units = self.units_selector.currentText()

        # Range
        try:
            vmin = float(self.vmin_input.text()) if self.vmin_input.text() else None
            vmax = float(self.vmax_input.text()) if self.vmax_input.text() else None
        except ValueError:
            vmin = vmax = None

        for cap in subfolders:
            cap_path = os.path.join(self.batch_root, cap)
            # find single CSV in this subfolder
            csvs = glob.glob(os.path.join(cap_path, "*.csv"))
            if len(csvs) != 1:
                failures.append(f"{cap}: expected exactly 1 CSV, found {len(csvs)}")
                continue

            try:
                primary = np.loadtxt(csvs[0], delimiter=',')
                if primary.shape != (10, 10):
                    raise ValueError("primary CSV is not 10x10.")

                # Apply operation
                result = primary.copy()
                if op == 'subtract':
                    if sub_ref is None:
                        # No reference provided — treat as no-op
                        pass
                    else:
                        if sub_ref.shape != result.shape:
                            raise ValueError("subtract ref shape mismatch.")
                        result = result - sub_ref
                        result[result == 0] = 1e-10
                elif op == 'divide':
                    if div_ref is None:
                        pass
                    else:
                        if div_ref.shape != result.shape:
                            raise ValueError("divide ref shape mismatch.")
                        with np.errstate(divide='ignore', invalid='ignore'):
                            result = np.true_divide(result, div_ref)
                            result[~np.isfinite(result)] = 0.0
                        result[result == 0] = 1e-10

                # Units conversion unless ratio
                if op == 'divide':
                    converted = result
                else:
                    converted = result * self.get_unit_conversion_factor(units)

                # Title per capture
                title = f"{title_base} [{cap}]" if title_base else cap

                # Render and save
                self.canvas.plot(
                    converted, cmap=cmap, vmin=vmin, vmax=vmax, log_scale=log_scale,
                    title=title, colorbar_label=cblabel, show_values=show_values,
                    bad_pixel_coords=self.bad_pixel_coords, bad_pixel_color=self.bad_pixel_color
                )
                out_path = os.path.join(out_dir, f"{cap}.png")
                self.canvas.fig.savefig(out_path, dpi=300)
                processed += 1

            except Exception as e:
                failures.append(f"{cap}: {e}")

        msg = f"Processed: {processed}\nOutput: {out_dir}"
        if failures:
            msg += "\n\nFailures:\n- " + "\n- ".join(failures)
        QMessageBox.information(self, "Batch Complete", msg)

# ------------------------------ Entrypoint ------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
