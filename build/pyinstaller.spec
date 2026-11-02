# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

app_root = os.path.abspath(os.path.join(SPECPATH, ".."))

hiddenimports = []
hiddenimports += collect_submodules("geopandas")
hiddenimports += collect_submodules("fiona")
hiddenimports += collect_submodules("pyproj")
hiddenimports += collect_submodules("shapely")
hiddenimports += collect_submodules("matplotlib.backends")
hiddenimports += collect_submodules("rasterio")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("contextily")
hiddenimports += collect_submodules("PySide6")

datas = []
datas += collect_data_files("matplotlib")
datas += collect_data_files("geopandas")
datas += collect_data_files("fiona")
datas += collect_data_files("pyproj")
datas += collect_data_files("rasterio")
datas += collect_data_files("scipy")
datas += collect_data_files("contextily")
datas += collect_data_files("PySide6")
datas += [(os.path.join(app_root, "assets"), "assets")]

binaries = []
binaries += collect_dynamic_libs("rasterio")
binaries += collect_dynamic_libs("fiona")
binaries += collect_dynamic_libs("pyproj")
binaries += collect_dynamic_libs("shapely")
binaries += collect_dynamic_libs("scipy")

a = Analysis(
    [os.path.join(app_root, "main.py")],
    pathex=[app_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="InSARLandslideGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="InSARLandslideGUI.app",
    icon=None,
    bundle_identifier="org.opensra2.landslide.gui",
)
