import pyvisa
from serial.tools import list_ports

try:  # Prefer PyQt5 to match the rest of the app; fall back to PySide6 otherwise.
    from PyQt5 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - fallback path
    from PySide6 import QtCore, QtWidgets


def _scan_visa_resources():
    items = []
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


class DevicePicker(QtWidgets.QDialog):
    def __init__(self, parent, title="Select device", show_gpib=False, default_gpib=5):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.combo = QtWidgets.QComboBox(self)
        self.combo.setMinimumWidth(520)
        self.refresh_btn = QtWidgets.QPushButton("Rescan")
        self.manual = QtWidgets.QLineEdit(self)
        self.manual.setPlaceholderText(
            "Or paste a resource/port (e.g. USB0::..., GPIB0::..., COM3, /dev/ttyACM0, 192.168.1.50)"
        )
        self.manual.setClearButtonEnabled(True)

        gpw = QtWidgets.QWidget(self)
        gpl = QtWidgets.QHBoxLayout(gpw)
        gpl.setContentsMargins(0, 0, 0, 0)
        self.gpib_label = QtWidgets.QLabel("GPIB address (Prologix):")
        self.gpib_spin = QtWidgets.QSpinBox(self)
        self.gpib_spin.setRange(0, 30)
        self.gpib_spin.setValue(default_gpib)
        gpl.addWidget(self.gpib_label)
        gpl.addWidget(self.gpib_spin)
        gpl.addStretch(1)
        gpw.setVisible(show_gpib)
        self._gpw = gpw

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Detected devices:"))
        lay.addWidget(self.combo)
        lay.addWidget(self.refresh_btn)
        lay.addSpacing(8)
        lay.addWidget(QtWidgets.QLabel("Advanced:"))
        lay.addWidget(self.manual)
        lay.addWidget(gpw)
        lay.addWidget(btns)

        self._items = []
        self.refresh_btn.clicked.connect(self._populate)
        self._populate()

    def _populate(self):
        progress = QtWidgets.QProgressDialog("Scanning devices…", None, 0, 0, self)
        progress.setWindowTitle("Scanning devices")
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        progress.show()
        QtWidgets.QApplication.processEvents()
        self.combo.clear()
        self._items = []
        try:
            visa_items = _scan_visa_resources()
            serial_items = _scan_serial_ports()
        finally:
            progress.close()
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

    def get(self):
        manual = self.manual.text().strip()
        if manual:
            return manual, (self.gpib_spin.value() if self._gpw.isVisible() else None)
        idx = self.combo.currentIndex()
        if idx < 0 or idx >= len(self._items):
            return None, None
        item = self._items[idx]
        if item.get("kind") in ("header", None):
            return None, None
        return item["value"], (
            self.gpib_spin.value() if self._gpw.isVisible() else None
        )
