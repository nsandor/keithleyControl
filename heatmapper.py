import sys
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

class HeatmapCanvas(FigureCanvas):
    """A canvas that updates the plot."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create the figure first
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Correctly initialize the parent class with the figure object
        super().__init__(self.fig)
        
        # Now add the subplot and set the parent
        self.axes = self.fig.add_subplot(111)
        self.setParent(parent)
        
        # Initialize instance variables
        self.image = None
        self.colorbar = None # Reference to the colorbar
        
    def plot(self, data, cmap='viridis', vmin=None, vmax=None, log_scale=False, 
             title="", colorbar_label="", show_values=False,
             bad_pixel_coords=None, bad_pixel_color=None):
        """
        Plots the heatmap on the canvas, correctly handling the colorbar.
        This version clears the entire figure to prevent state issues with the colorbar.
        """
        # Clear the entire figure to ensure a clean slate, which is more robust
        # than trying to manage the state of individual artists like the colorbar.
        self.fig.clear()
        
        # Re-create the axes on the cleared figure
        self.axes = self.fig.add_subplot(111)
        
        # Reset references
        self.image = None
        self.colorbar = None
        
        if data is not None:
            if log_scale:
                # Shift data to be non-negative before taking the log
                min_val = np.min(data)
                data_to_plot = np.log1p(data - min_val)
            else:
                data_to_plot = data
                
            if log_scale:
                self.image = self.axes.imshow(data_to_plot, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest')
            else:
                self.image = self.axes.imshow(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Create a new colorbar and store a reference to it
            self.colorbar = self.fig.colorbar(self.image, ax=self.axes)
            
            # Set colorbar label if provided
            if colorbar_label:
                self.colorbar.set_label(colorbar_label)
            
            # Set title if provided
            if title:
                self.axes.set_title(title)
            
            # Overlay bad pixels if provided
            if bad_pixel_coords and bad_pixel_color:
                rgba = to_rgba(bad_pixel_color, alpha=0.85)
                for (r, c) in bad_pixel_coords:
                    # Draw a rectangle over the pixel. imshow cells are centered on integer coords.
                    rect = Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=rgba, edgecolor='black', linewidth=0.5)
                    self.axes.add_patch(rect)
            
            # Show data values on each pixel if requested
            if show_values and data is not None:
                height, width = data.shape
                for i in range(height):
                    for j in range(width):
                        value = data[i, j]
                        # Format the value to avoid too many decimal places
                        if abs(value) < 1000:
                            text = f'{value:.2f}'
                        else:
                            text = f'{value:.1e}'
                        self.axes.text(j, i, text, ha='center', va='center', 
                                     color='white',
                                     fontsize=8, weight='bold')
        
        # Redraw the canvas to show changes
        self.draw()
        
    def add_annotation(self, x, y, text, color='black'):
        """Adds a text annotation to the plot."""
        if self.image:
            self.axes.text(x, y, text, color=color, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.5, pad=2))
            self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photodiode Data Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.datasets = {}
        self.current_data_key = None
        self.subtract_data_key = None
        self.divide_data_key = None
        self.result_data = None
        self.operation = None  # None | 'subtract' | 'divide'
        
        # --- Member for annotation event handling ---
        self.annotation_cid = None
        
        # Bad pixel handling
        self.bad_pixel_coords = []  # list of (row,col)
        self.bad_pixel_color = '#FF0000'
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(320)
        
        # File Operations Group
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
        
        # Data Selection Group
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
        
        # Plot Appearance Group
        appearance_group = QGroupBox("Plot Appearance")
        appearance_layout = QFormLayout()
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Enter plot title")
        self.colorbar_label_input = QLineEdit()
        self.colorbar_label_input.setPlaceholderText("Enter colorbar label (e.g., 'Current')")
        self.units_selector = QComboBox()
        self.units_selector.addItems(['pA', 'nA', 'µA', 'mA'])
        self.units_selector.setCurrentText('µA')  # Default to µA (common for photodiode measurements)
        self.show_values_toggle = QPushButton("Show Values: OFF")
        self.show_values_toggle.setCheckable(True)
        appearance_layout.addRow(QLabel("Plot Title:"), self.title_input)
        appearance_layout.addRow(QLabel("Colorbar Label:"), self.colorbar_label_input)
        appearance_layout.addRow(QLabel("Units:"), self.units_selector)
        appearance_layout.addRow(self.show_values_toggle)
        appearance_group.setLayout(appearance_layout)
        left_layout.addWidget(appearance_group)
        
        # Heatmap Controls Group
        heatmap_group = QGroupBox("Heatmap Controls")
        heatmap_layout = QFormLayout()
        self.cmap_selector = QComboBox()
        self.cmap_selector.addItems(plt.colormaps())
        self.vmin_input = QLineEdit()
        self.vmax_input = QLineEdit()
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
        
        # Bad Pixels Group
        bad_group = QGroupBox("Bad Pixels")
        bad_layout = QFormLayout()
        self.bad_indices_input = QLineEdit()
        self.bad_indices_input.setPlaceholderText("e.g. 1,5,12-15,87")
        self.bad_color_button = QPushButton("Select Color")
        self.bad_color_display = QLabel()
        self.bad_color_display.setFixedHeight(22)
        self.bad_color_display.setStyleSheet(f"background-color: {self.bad_pixel_color}; border: 1px solid #444;")
        bad_layout.addRow(QLabel("Indices:"), self.bad_indices_input)
        bad_layout.addRow(self.bad_color_button, self.bad_color_display)
        bad_group.setLayout(bad_layout)
        left_layout.addWidget(bad_group)
        
        # Annotation Group
        annotation_group = QGroupBox("Annotations")
        annotation_layout = QFormLayout()
        self.add_text_button = QPushButton("Add Text Annotation")
        annotation_layout.addRow(self.add_text_button)
        annotation_group.setLayout(annotation_layout)
        left_layout.addWidget(annotation_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # --- Right Panel (Plot) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.canvas = HeatmapCanvas(self, width=8, height=6, dpi=100)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_panel)
        
        # --- Connect Signals and Slots ---
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
        
        self.update_selectors()
        
    # -------------------------- Bad Pixels Handling --------------------------
    def choose_bad_pixel_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bad_pixel_color = color.name()
            self.bad_color_display.setStyleSheet(f"background-color: {self.bad_pixel_color}; border: 1px solid #444;")
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
            except ValueError as e:
                # Show a non-blocking warning maybe later; for now message box only on explicit errors
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
                start, end = parts
                start_i = int(start)
                end_i = int(end)
                if start_i > end_i:
                    start_i, end_i = end_i, start_i
                result.extend(range(start_i, end_i + 1))
            else:
                result.append(int(tok))
        return result
    # -------------------------------------------------------------------------
        
    def import_csv(self):
        """Import a 10x10 CSV file."""
        path, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                data = np.loadtxt(path, delimiter=',')
                if data.shape != (10, 10):
                    raise ValueError("Data is not 10x10.")
                
                filename = path.split('/')[-1]
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
        """Export the current heatmap view as a PNG image."""
        if self.current_data_key is None and self.result_data is None:
            QMessageBox.warning(self, "Export Error", "No plot to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Export Image", "heatmap.png", "PNG Files (*.png);;JPEG Files (*.jpg)")
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
        
    def select_primary_data(self, index):
        """Handle selection of the primary dataset."""
        if index >= 0:
            self.current_data_key = self.data_selector.itemText(index)
            self.result_data = None # Clear operation result
            self.operation = None
            self.update_plot()
            
    def select_subtract_data(self, index):
        """Handle selection of the dataset to subtract."""
        if index > 0: # Index 0 is "None"
            self.subtract_data_key = self.subtract_selector.itemText(index)
        else:
            self.subtract_data_key = None
            
    def select_divide_data(self, index):
        """Handle selection of the dataset to divide by."""
        if index > 0: # Index 0 is "None"
            self.divide_data_key = self.divide_selector.itemText(index)
        else:
            self.divide_data_key = None
            
    def perform_subtraction(self):
        """Subtract one dataset from another."""
        if self.current_data_key and self.subtract_data_key:
            primary_data = self.datasets[self.current_data_key]
            subtract_data = self.datasets[self.subtract_data_key]
            if primary_data.shape != subtract_data.shape:
                QMessageBox.warning(self, "Subtraction Error", "Datasets must have the same shape.")
                return
            self.result_data = primary_data - subtract_data
            # Avoid exact zeros to be log-safe
            self.result_data[self.result_data == 0] = 1e-10
            self.operation = 'subtract'
            self.update_plot()
        else:
            QMessageBox.warning(self, "Subtraction Error", "Please select both a primary and a subtract dataset.")
            
    def perform_division(self):
        """Divide one dataset by another (element-wise)."""
        if self.current_data_key and self.divide_data_key:
            primary_data = self.datasets[self.current_data_key]
            denom_data = self.datasets[self.divide_data_key]
            if primary_data.shape != denom_data.shape:
                QMessageBox.warning(self, "Division Error", "Datasets must have the same shape.")
                return
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.true_divide(primary_data, denom_data)
                # Replace inf/NaN from division by zero with 0 for display stability
                result[~np.isfinite(result)] = 0.0
            # Avoid exact zeros to be log-safe if user enables log scale
            result[result == 0] = 1e-10
            self.result_data = result
            self.operation = 'divide'
            self.update_plot()
        else:
            QMessageBox.warning(self, "Division Error", "Please select both a primary and a divide dataset.")
            
    def clear_subtraction(self):
        """Clear the subtraction result and show the primary data."""
        self.result_data = None
        self.operation = None
        self.subtract_selector.setCurrentIndex(0)
        self.update_plot()
        
    def clear_division(self):
        """Clear the division result and show the primary data."""
        self.result_data = None
        self.operation = None
        self.divide_selector.setCurrentIndex(0)
        self.update_plot()
        
    def toggle_log_scale(self):
        """Toggle logarithmic scaling for the colormap."""
        if self.log_scale_toggle.isChecked():
            self.log_scale_toggle.setText("Toggle Log Scale: ON")
        else:
            self.log_scale_toggle.setText("Toggle Log Scale: OFF")
        self.update_plot()
        
    def get_unit_conversion_factor(self, target_unit):
        """Get conversion factor to convert from base unit (Amps) to target unit."""
        conversions = {
            'pA': 1e12,     # 1 A = 1,000,000,000,000 pA
            'nA': 1e9,      # 1 A = 1,000,000,000 nA
            'µA': 1e6,      # 1 A = 1,000,000 µA
            'mA': 1e3       # 1 A = 1,000 mA
        }
        return conversions.get(target_unit, 1)
    
    def convert_data_units(self, data, target_unit):
        """Convert data from base unit (Amps) to target unit."""
        factor = self.get_unit_conversion_factor(target_unit)
        return data * factor
    
    def get_colorbar_label(self):
        """Generate appropriate colorbar label with units."""
        base_label = self.colorbar_label_input.text()
        units = self.units_selector.currentText()
        
        # Division yields a unitless ratio
        if self.operation == 'divide':
            if base_label:
                return f"{base_label} (ratio)"
            else:
                return "Ratio"
        
        if base_label:
            return f"{base_label} ({units})"
        else:
            return f"Current ({units})"
    
    def toggle_show_values(self):
        """Toggle showing data values on each pixel."""
        if self.show_values_toggle.isChecked():
            self.show_values_toggle.setText("Show Values: ON")
        else:
            self.show_values_toggle.setText("Show Values: OFF")
        self.update_plot()
        
    def add_annotation(self):
        """Add a text annotation to the plot."""
        if self.current_data_key is None and self.result_data is None:
            QMessageBox.warning(self, "Annotation Error", "Please load data first.")
            return
            
        text, ok = QInputDialog.getText(self, 'Add Annotation', 'Enter text:')
        if ok and text:
            color = QColorDialog.getColor()
            if color.isValid():
                QMessageBox.information(self, "Place Annotation", "Click on the plot to place the text.")
                
                # Disconnect any previously connected handler to avoid duplicates
                if self.annotation_cid:
                    self.canvas.figure.canvas.mpl_disconnect(self.annotation_cid)
                
                # Connect the event and store the connection id (cid)
                self.annotation_cid = self.canvas.figure.canvas.mpl_connect(
                    'button_press_event', 
                    lambda event: self._place_annotation(event, text, color.name())
                )
                
    def _place_annotation(self, event, text, color_name):
        """Callback to place the annotation after a click."""
        # Check if the click was on the main axes
        if event.inaxes == self.canvas.axes:
            # Get integer coordinates for the cell
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.canvas.add_annotation(x, y, text, color=color_name)
            
            # Disconnect the handler now that annotation is placed
            if self.annotation_cid:
                self.canvas.figure.canvas.mpl_disconnect(self.annotation_cid)
                self.annotation_cid = None
                
    def update_plot(self):
        """Redraw the heatmap with current settings."""
        data_to_plot = None
        if self.result_data is not None:
            data_to_plot = self.result_data
        elif self.current_data_key:
            data_to_plot = self.datasets[self.current_data_key]
            
        if data_to_plot is not None:
            # Convert data to selected units unless we're showing a unitless ratio
            selected_units = self.units_selector.currentText()
            if self.operation == 'divide':
                converted_data = data_to_plot  # unitless ratio
            else:
                converted_data = self.convert_data_units(data_to_plot, selected_units)
            
            cmap = self.cmap_selector.currentText()
            try:
                vmin = float(self.vmin_input.text()) if self.vmin_input.text() else None
                vmax = float(self.vmax_input.text()) if self.vmax_input.text() else None
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Invalid range. Please enter numbers only.")
                vmin, vmax = None, None
            
            log_scale = self.log_scale_toggle.isChecked()
            title = self.title_input.text()
            colorbar_label = self.get_colorbar_label()
            show_values = self.show_values_toggle.isChecked()
            
            self.canvas.plot(converted_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                           log_scale=log_scale, title=title, 
                           colorbar_label=colorbar_label, show_values=show_values,
                           bad_pixel_coords=self.bad_pixel_coords, bad_pixel_color=self.bad_pixel_color)
        else:
            self.canvas.plot(None) # Clear canvas if no data

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
