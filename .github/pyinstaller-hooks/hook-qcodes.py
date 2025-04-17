# .github/pyinstaller-hooks/hook-qcodes.py
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("qcodes")
