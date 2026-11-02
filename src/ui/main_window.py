from __future__ import annotations

import json
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.paths import program_dir, resource_dir
from src.core.workflow import (
    export_state,
    run_step1_1,
    run_step1_2,
    run_step1_3,
    run_step1,
    run_step2_1,
    run_step2_2,
    run_step2_3,
    run_step2,
    run_step3_pick,
    run_step3_mesh,
    run_step3_response,
    run_step3,
    run_steps_1_and_2,
)
from src.ui.contour_picker import run_contour_picker
from src.ui.plotting import MplTab, render_plots

_mpl_cache = program_dir() / ".mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


class PipelineWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, config: AppConfig, step_fn, state: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.step_fn = step_fn
        self.state = state or {}
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            def _progress(pct, msg):
                if self._stop:
                    raise InterruptedError("Computation stopped by user.")
                self.progress.emit(pct, msg)

            result = self.step_fn(self.config, self.state, progress=_progress)
            self.finished.emit(result)
        except InterruptedError as exc:
            self.failed.emit(f"CANCELLED:{exc}")
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InSAR-Landslide-Pipeline GUI")
        self.resize(1620, 950)
        self.config = AppConfig()
        self.pipeline_state: Optional[Dict] = None
        self.thread = None
        self.worker = None
        self.current_task = ""
        self.selected_polygon_override = None
        self._overlay_cache = {"pipe": None, "fault": None}
        self._cbars: Dict = {}
        self.inputs: Dict[str, QLineEdit] = {}
        self.params: Dict[str, QWidget] = {}
        self.param_labels = self._build_param_label_map()
        self._build_ui()
        self._fill_from_config(self.config)
        self._connect_live_plot_params()

    def _build_param_label_map(self) -> Dict[str, str]:
        return {
            "date_match_max_days": "Max Date Gap (days)",
            "spatial_match_max_m": "Max Point Spacing (m)",
            "dst_epsg": "Target EPSG",
            "sigma_px": "Gaussian Sigma (pixels)",
            "dem_resolution_m": "DEM Resolution (m)",
            "resampling_method": "DEM Resampling",
            "asc_heading_deg": "Ascending Heading (deg)",
            "asc_inc_deg": "Ascending Incidence (deg)",
            "desc_heading_deg": "Descending Heading (deg)",
            "desc_inc_deg": "Descending Incidence (deg)",
            "flip_constraint_sign": "Reverse Slope Constraint",
            "condition_threshold": "Max Condition Number",
            "day_idx": "Date Index",
            "dir_idx": "Contour Direction (0=N, 1=E, 2=Z)",
            "contour_threshold": "Displacement Threshold (mm)",
            "filter_mode": "Threshold Side",
            "k_neighbors": "Smooth Neighbor Count",
            "top_n": "Keep Longest N Pipelines",
            "k_fallback": "Fallback Neighbor Count",
            "field_smooth_enable": "Smooth Displacement",
            "gauss_bandwidth_m": "Smooth Radius (m)",
            "mesh_size_m": "Mesh Spacing (m)",
            "pipeline_diameter_m": "Default Diameter (m)",
            "selected_lines_shp_name": "Contour GIS Filename",
            "annualize": "Annualize Displacement/Strain",
            "annualize_span_days": "Annualize Span (days)",
            "save_intermediate": "Save Intermediate Files",
            "date_column_prefix": "Date Column Prefix",
        }

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        body_pt = self.font().pointSize()
        if body_pt <= 0:
            body_pt = 13
        title_pt = body_pt + 1
        self.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                font-size: {title_pt}pt;
            }}
            QGroupBox QLabel,
            QGroupBox QLineEdit,
            QGroupBox QComboBox,
            QGroupBox QPushButton,
            QGroupBox QPlainTextEdit,
            QGroupBox QTabBar {{
                font-weight: normal;
                font-size: {body_pt}pt;
            }}
            QGroupBox#stepGroup {{
                font-weight: normal;
                font-style: italic;
                font-size: {title_pt}pt;
            }}
            QGroupBox#stepGroup QPushButton {{
                font-weight: normal;
                font-style: normal;
                font-size: {body_pt}pt;
            }}
            QGroupBox#visGroup {{
                font-weight: bold;
                font-size: {title_pt}pt;
                padding-top: 16px;
                margin-top: 4px;
            }}
            QGroupBox#visGroup::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px 10px 0;
            }}
            """
        )
        v = QVBoxLayout(root)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)
        v.addWidget(splitter, stretch=1)

        left = QWidget()
        left.setMinimumWidth(390)
        ll = QVBoxLayout(left)
        ll.addWidget(self._build_file_group())
        ll.addWidget(self._build_run_group())
        ll.addStretch(1)
        ll.addWidget(self._build_logo_panel())
        splitter.addWidget(left)

        middle_scroll = QScrollArea()
        middle_scroll.setWidgetResizable(True)
        middle_scroll.setMinimumWidth(456)
        middle_holder = QWidget()
        ml = QVBoxLayout(middle_holder)
        ml.addWidget(self._build_param_groups())
        ml.addStretch(1)
        middle_scroll.setWidget(middle_holder)
        splitter.addWidget(middle_scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        box = QGroupBox("Visualization")
        box.setObjectName("visGroup")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(14, 20, 14, 10)
        bl.setSpacing(8)
        self._init_vis_controls()
        self.tabs = QTabWidget()
        self.tab_dem = MplTab("DEM & Aspect")
        self.tab_insar = MplTab("InSAR map")
        self.tab_mesh = MplTab("Mesh")
        self.tab_pipe_age = MplTab("Pipeline age")
        self.tab_pipe_mat = MplTab("Pipeline material")
        self.tab_pipe_diam = MplTab("Pipeline diameters")
        self.tab_disp = MplTab("Pipeline displacement")
        self.tab_strain = MplTab("Pipeline strain")
        self._vis_plot_tabs = [
            self.tab_dem,
            self.tab_insar,
            self.tab_mesh,
            self.tab_pipe_age,
            self.tab_pipe_mat,
            self.tab_pipe_diam,
            self.tab_disp,
            self.tab_strain,
        ]
        self._vis_param_hosts = []
        tab_titles = [
            "DEM && Aspect",
            "InSAR map",
            "Mesh",
            "Pipeline age",
            "Pipeline material",
            "Pipeline diameters",
            "Pipeline displacement",
            "Pipeline strain",
        ]
        for plot, title in zip(self._vis_plot_tabs, tab_titles):
            self.tabs.addTab(self._wrap_vis_tab(plot), title)
        self.tabs.currentChanged.connect(self._sync_vis_param_bar)
        self._sync_vis_param_bar(self.tabs.currentIndex())
        bl.addWidget(self.tabs)
        self.btn_open_map_popup = QPushButton("Open in New Window")
        self.btn_open_map_popup.clicked.connect(self.on_open_map_popup)
        bl.addWidget(self.btn_open_map_popup)
        log_wrap = QWidget()
        log_lay = QVBoxLayout(log_wrap)
        log_head = QHBoxLayout()
        log_title = QLabel("Log")
        log_title.setStyleSheet("font-weight: bold;")
        log_head.addWidget(log_title)
        log_head.addStretch(1)
        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.clicked.connect(self.on_clear_log)
        log_head.addWidget(self.btn_clear_log)
        log_lay.addLayout(log_head)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        log_lay.addWidget(self.log)
        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(box)
        right_split.addWidget(log_wrap)
        right_split.setSizes([620, 220])
        rl.addWidget(right_split)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([390, 456, 774])
        self._build_menu()

    def _vis_labeled(self, text: str, *widgets) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(QLabel(text))
        for w in widgets:
            lay.addWidget(w)
        return box

    def _make_vis_param_host(self) -> QWidget:
        host = QWidget()
        host.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 4)
        v.setSpacing(2)
        row1 = QWidget()
        r1 = QHBoxLayout(row1)
        r1.setContentsMargins(0, 0, 0, 0)
        r1.setSpacing(6)
        row2 = QWidget()
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.setSpacing(6)
        v.addWidget(row1)
        v.addWidget(row2)
        host._row1 = row1
        host._row2 = row2
        return host

    def _wrap_vis_tab(self, plot: QWidget) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)
        host = self._make_vis_param_host()
        self._vis_param_hosts.append(host)
        lay.addWidget(host)
        lay.addWidget(plot, 1)
        return page

    def _init_vis_controls(self) -> None:
        self._vis_param_pool = QWidget(self)
        self._vis_param_pool.hide()

        self.cmb_epoch = QComboBox()
        self.cmb_epoch.addItem("Run Step2 to load epochs", -1)
        self.cmb_epoch.currentIndexChanged.connect(self.on_map_control_changed)
        self.cmb_epoch.setEnabled(False)
        self.cmb_nez_dir = QComboBox()
        self.cmb_nez_dir.addItems(["N", "E", "Z"])
        self.cmb_nez_dir.setCurrentText("E")
        self.cmb_nez_dir.currentIndexChanged.connect(self.on_map_control_changed)
        self.cmb_insar_field = QComboBox()
        self.cmb_insar_field.addItems(["Complete field", "Filtered field"])
        self.cmb_insar_field.setCurrentText("Complete field")
        self.cmb_insar_field.currentIndexChanged.connect(self.on_map_control_changed)
        self.le_vmin = QLineEdit("-80")
        self.le_vmin.setPlaceholderText("auto")
        self.le_vmin.setFixedWidth(80)
        self.le_vmin.editingFinished.connect(self.on_map_control_changed)
        self.le_vmax = QLineEdit("80")
        self.le_vmax.setPlaceholderText("auto")
        self.le_vmax.setFixedWidth(80)
        self.le_vmax.editingFinished.connect(self.on_map_control_changed)
        self.cmb_aspect_view = QComboBox()
        self.cmb_aspect_view.addItems(["Smoothed aspect", "Raw aspect"])
        self.cmb_aspect_view.currentIndexChanged.connect(self.render_plots)
        self.le_xlim_min = QLineEdit("")
        self.le_xlim_min.setPlaceholderText("auto")
        self.le_xlim_min.setFixedWidth(90)
        self.le_xlim_min.editingFinished.connect(self.on_map_control_changed)
        self.le_xlim_max = QLineEdit("")
        self.le_xlim_max.setPlaceholderText("auto")
        self.le_xlim_max.setFixedWidth(90)
        self.le_xlim_max.editingFinished.connect(self.on_map_control_changed)
        self.le_ylim_min = QLineEdit("")
        self.le_ylim_min.setPlaceholderText("auto")
        self.le_ylim_min.setFixedWidth(90)
        self.le_ylim_min.editingFinished.connect(self.on_map_control_changed)
        self.le_ylim_max = QLineEdit("")
        self.le_ylim_max.setPlaceholderText("auto")
        self.le_ylim_max.setFixedWidth(90)
        self.le_ylim_max.editingFinished.connect(self.on_map_control_changed)
        self.cmb_strain_type = QComboBox()
        self.cmb_strain_type.addItems(["Axial", "Bending"])
        self.cmb_strain_type.currentIndexChanged.connect(self.on_map_control_changed)
        self.cmb_strain_period = QComboBox()
        self.cmb_strain_period.addItems(["Data period", "Yearly", "Entire life"])
        self.cmb_strain_period.setCurrentText("Data period")
        self.cmb_strain_period.currentIndexChanged.connect(self.on_map_control_changed)
        self.cmb_use_diam_field = QComboBox()
        self.cmb_use_diam_field.addItems(["False", "True"])
        self.cmb_use_diam_field.setCurrentText("False")
        self.cmb_use_diam_field.currentIndexChanged.connect(self.on_map_control_changed)

        self._vis_grp_date = self._vis_labeled("Date", self.cmb_epoch)
        self._vis_grp_nez = self._vis_labeled("NEZ", self.cmb_nez_dir)
        self._vis_grp_field = self._vis_labeled("Field", self.cmb_insar_field)
        self._vis_grp_vmin = self._vis_labeled("vmin", self.le_vmin)
        self._vis_grp_vmax = self._vis_labeled("vmax", self.le_vmax)
        self._vis_grp_aspect = self._vis_labeled("Aspect view:", self.cmb_aspect_view)
        self._vis_grp_xlim = self._vis_labeled("map xlim", self.le_xlim_min, self.le_xlim_max)
        self._vis_grp_ylim = self._vis_labeled("map ylim", self.le_ylim_min, self.le_ylim_max)
        self._vis_grp_strain_type = self._vis_labeled("Strain type", self.cmb_strain_type)
        self._vis_grp_strain_period = self._vis_labeled("Strain period", self.cmb_strain_period)
        self._vis_grp_diam = self._vis_labeled("Consider pipeline diameter field", self.cmb_use_diam_field)
        self._vis_param_groups = [
            self._vis_grp_date,
            self._vis_grp_nez,
            self._vis_grp_field,
            self._vis_grp_vmin,
            self._vis_grp_vmax,
            self._vis_grp_aspect,
            self._vis_grp_xlim,
            self._vis_grp_ylim,
            self._vis_grp_strain_type,
            self._vis_grp_strain_period,
            self._vis_grp_diam,
        ]
        for grp in self._vis_param_groups:
            grp.setParent(self._vis_param_pool)

    def _fill_vis_param_row(self, row_widget: QWidget, groups) -> None:
        lay = row_widget.layout()
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self._vis_param_pool)
        if not groups:
            row_widget.hide()
            return
        for g in groups:
            lay.addWidget(g)
        lay.addStretch(1)
        row_widget.show()

    def _sync_vis_param_bar(self, index: int) -> None:
        if not getattr(self, "_vis_param_hosts", None):
            return
        for host in self._vis_param_hosts:
            self._fill_vis_param_row(host._row1, ())
            self._fill_vis_param_row(host._row2, ())
        groups_by_tab = {
            0: [(self._vis_grp_aspect,), ()],
            1: [
                (self._vis_grp_date, self._vis_grp_nez, self._vis_grp_field, self._vis_grp_vmin, self._vis_grp_vmax),
                (self._vis_grp_xlim, self._vis_grp_ylim),
            ],
            2: [(self._vis_grp_xlim, self._vis_grp_ylim), ()],
            3: [(self._vis_grp_xlim, self._vis_grp_ylim), ()],
            4: [(self._vis_grp_xlim, self._vis_grp_ylim), ()],
            5: [(self._vis_grp_xlim, self._vis_grp_ylim), ()],
            6: [(self._vis_grp_nez,), ()],
            7: [
                (self._vis_grp_strain_type, self._vis_grp_strain_period, self._vis_grp_diam),
                (self._vis_grp_xlim, self._vis_grp_ylim),
            ],
        }
        if index < 0 or index >= len(self._vis_param_hosts):
            return
        row1, row2 = groups_by_tab.get(index, ((), ()))
        host = self._vis_param_hosts[index]
        self._fill_vis_param_row(host._row1, row1)
        self._fill_vis_param_row(host._row2, row2)

    def _build_logo_panel(self):
        box = QGroupBox("")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)
        assets = resource_dir() / "assets"
        logo_candidates = [
            assets / "UCB logo.jpg",
            assets / "opensra2_logo.png",
        ]
        for p in logo_candidates:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            pix = QPixmap(str(p))
            if pix.isNull():
                lbl.setText(f"Missing logo: {p.name}")
            else:
                lbl.setPixmap(pix.scaled(165, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lay.addWidget(lbl)
        return box

    def _build_menu(self):
        menu = self.menuBar().addMenu("Tools")
        act_reset = QAction("Reset defaults", self)
        act_reset.triggered.connect(self.reset_defaults)
        menu.addAction(act_reset)
        act_save = QAction("Save parameter preset", self)
        act_save.triggered.connect(self.save_preset)
        menu.addAction(act_save)
        act_load = QAction("Load parameter preset", self)
        act_load.triggered.connect(self.load_preset)
        menu.addAction(act_load)
        act_save_state = QAction("Save workflow state", self)
        act_save_state.triggered.connect(self.save_workflow_state)
        menu.addAction(act_save_state)
        act_load_state = QAction("Load workflow state", self)
        act_load_state.triggered.connect(self.load_workflow_state)
        menu.addAction(act_load_state)

    def _build_file_group(self):
        box = QGroupBox("Input Files")
        form = QFormLayout(box)
        form.setContentsMargins(8, 6, 8, 6)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        file_fields = [
            ("dem_paths", "DEM TIFF files"),
            ("asc_shp_path", "Ascending InSAR shapefile"),
            ("desc_shp_path", "Descending InSAR shapefile"),
            ("fault_line_shp_path", "Fault line shapefile"),
            ("pipe_axis_path", "Pipe axis geometry"),
            ("output_dir", "Output directory"),
        ]
        for key, label in file_fields:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            edit = QLineEdit()
            edit.setMinimumHeight(24)
            btn = QPushButton("...")
            btn.setFixedWidth(36)
            btn.setMinimumHeight(24)
            btn.clicked.connect(lambda _, k=key: self._browse_file(k))
            lay.addWidget(edit)
            lay.addWidget(btn)
            self.inputs[key] = edit
            form.addRow(label, row)
        return box

    def _build_run_group(self):
        box = QGroupBox("Run Analysis")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        self.btn_step11 = QPushButton("Step 1-1: Load && Merge DEM")
        self.btn_step12 = QPushButton("Step 1-2: Compute Slope Aspect")
        self.btn_step13 = QPushButton("Step 1-3: Smooth DEM && Aspect")
        self.btn_step1 = QPushButton("Run All of Step 1")
        self.btn_step21 = QPushButton("Step 2-1: Match Asc/Desc InSAR")
        self.btn_step22 = QPushButton("Step 2-2: Compute N/E/Z Displacement")
        self.btn_step23 = QPushButton("Step 2-3: Filter Unstable Points")
        self.btn_step2 = QPushButton("Run All of Step 2")
        self.btn_step31 = QPushButton("Step 3-1: Select Landslide Area")
        self.btn_step32 = QPushButton("Step 3-2: Build Pipeline Mesh")
        self.btn_step33 = QPushButton("Step 3-3: Compute Displacement && Strain")
        self.btn_step3 = QPushButton("Run All of Step 3")
        self.btn_all = QPushButton("Run All Steps")
        self.btn_stop = QPushButton("Stop")
        self.btn_save_status = QPushButton("Save status")
        self.btn_load_status = QPushButton("Load status")
        self.btn_export = QPushButton("Export Results")
        self.btn_open = QPushButton("Open Output Folder")
        self.btn_stop.setEnabled(False)
        self.btn_save_status.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_open.setEnabled(False)

        self.btn_step11.clicked.connect(self.on_run_step11)
        self.btn_step12.clicked.connect(self.on_run_step12)
        self.btn_step13.clicked.connect(self.on_run_step13)
        self.btn_step1.clicked.connect(self.on_run_step1)
        self.btn_step21.clicked.connect(self.on_run_step21)
        self.btn_step22.clicked.connect(self.on_run_step22)
        self.btn_step23.clicked.connect(self.on_run_step23)
        self.btn_step31.clicked.connect(self.on_run_step31)
        self.btn_step32.clicked.connect(self.on_run_step32)
        self.btn_step33.clicked.connect(self.on_run_step33)
        self.btn_step2.clicked.connect(self.on_run_step2)
        self.btn_step3.clicked.connect(self.on_run_step3)
        self.btn_all.clicked.connect(self.on_run_all)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_save_status.clicked.connect(self.save_workflow_state)
        self.btn_load_status.clicked.connect(self.load_workflow_state)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_open.clicked.connect(self.on_open_output)

        for b in [
            self.btn_step11,
            self.btn_step12,
            self.btn_step13,
            self.btn_step1,
            self.btn_step21,
            self.btn_step22,
            self.btn_step23,
            self.btn_step2,
            self.btn_step31,
            self.btn_step32,
            self.btn_step33,
            self.btn_step3,
            self.btn_all,
            self.btn_stop,
            self.btn_save_status,
            self.btn_load_status,
            self.btn_export,
            self.btn_open,
        ]:
            b.setMinimumHeight(30)

        step1_box = QGroupBox("Step 1 (DEM Processing)")
        step1_box.setObjectName("stepGroup")
        step1_lay = QVBoxLayout(step1_box)
        step1_lay.setContentsMargins(8, 8, 8, 8)
        step1_lay.setSpacing(6)
        step1_lay.addWidget(self.btn_step11)
        step1_lay.addWidget(self.btn_step12)
        step1_lay.addWidget(self.btn_step13)
        step1_lay.addWidget(self.btn_step1)

        step2_box = QGroupBox("Step 2 (Ground Deformation Analysis)")
        step2_box.setObjectName("stepGroup")
        step2_lay = QVBoxLayout(step2_box)
        step2_lay.setContentsMargins(8, 8, 8, 8)
        step2_lay.setSpacing(6)
        step2_lay.addWidget(self.btn_step21)
        step2_lay.addWidget(self.btn_step22)
        step2_lay.addWidget(self.btn_step23)
        step2_lay.addWidget(self.btn_step2)

        step3_box = QGroupBox("Step 3 (Pipeline Disp. && Strain Estimation)")
        step3_box.setObjectName("stepGroup")
        step3_lay = QVBoxLayout(step3_box)
        step3_lay.setContentsMargins(8, 8, 8, 8)
        step3_lay.setSpacing(6)
        step3_lay.addWidget(self.btn_step31)
        step3_lay.addWidget(self.btn_step32)
        step3_lay.addWidget(self.btn_step33)
        step3_lay.addWidget(self.btn_step3)

        lay.addWidget(step1_box)
        lay.addWidget(step2_box)
        lay.addWidget(step3_box)
        lay.addWidget(self.btn_all)
        lay.addWidget(self.btn_stop)
        lay.addWidget(self.btn_save_status)
        lay.addWidget(self.btn_load_status)
        lay.addWidget(self.btn_export)
        lay.addWidget(self.btn_open)
        return box

    def _build_param_groups(self):
        holder = QWidget()
        lay = QVBoxLayout(holder)
        groups = [
            (
                "Step 1 DEM Processing",
                ["sigma_px", "dem_resolution_m", "resampling_method"],
            ),
            (
                "Step 2 InSAR Data Processing",
                [
                    "date_match_max_days",
                    "spatial_match_max_m",
                    "asc_heading_deg",
                    "asc_inc_deg",
                    "desc_heading_deg",
                    "desc_inc_deg",
                    "flip_constraint_sign",
                    "condition_threshold",
                ],
            ),
            (
                "Step 3 Contour && Strain Estimation",
                [
                    "day_idx",
                    "dir_idx",
                    "contour_threshold",
                    "filter_mode",
                    "field_smooth_enable",
                    "k_neighbors",
                    "gauss_bandwidth_m",
                    "k_fallback",
                    "top_n",
                    "mesh_size_m",
                    "pipeline_diameter_m",
                    "selected_lines_shp_name",
                ],
            ),
        ]
        for title, keys in groups:
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.DontWrapRows)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form.setHorizontalSpacing(10)
            for key in keys:
                form.addRow(self.param_labels.get(key, key), self._make_editor(key))
            lay.addWidget(box)
        return holder

    def _make_editor(self, key: str):
        value = getattr(self.config, key)
        if isinstance(value, bool):
            cmb = QComboBox()
            cmb.addItems(["True", "False"])
            self.params[key] = cmb
            return cmb
        if key == "resampling_method":
            cmb = QComboBox()
            cmb.addItems(["nearest", "bilinear", "cubic"])
            self.params[key] = cmb
            return cmb
        if key == "filter_mode":
            cmb = QComboBox()
            cmb.addItems(["below", "above"])
            self.params[key] = cmb
            return cmb
        e = QLineEdit()
        self.params[key] = e
        return e

    def _connect_live_plot_params(self):
        for key in ("contour_threshold", "filter_mode", "pipeline_diameter_m", "day_idx", "dir_idx"):
            w = self.params.get(key)
            if w is None:
                continue
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.on_map_control_changed)
            else:
                w.editingFinished.connect(self.on_map_control_changed)

    def _browse_file(self, key: str):
        if key == "dem_paths":
            files, _ = QFileDialog.getOpenFileNames(self, "Select two DEM tiff files", "", "TIFF (*.tif *.tiff)")
            if files:
                self.inputs[key].setText(";".join(files[:2]))
            return
        if key == "output_dir":
            p = QFileDialog.getExistingDirectory(self, "Select output directory")
        else:
            p, _ = QFileDialog.getOpenFileName(self, f"Select {key}")
        if p:
            self.inputs[key].setText(p)

    def _fill_from_config(self, config: AppConfig):
        for k, w in self.inputs.items():
            w.setText(str(getattr(config, k)))
        for k, w in self.params.items():
            val = getattr(config, k)
            if isinstance(w, QComboBox):
                w.setCurrentText(str(val))
            else:
                w.setText(str(val))

    def _read_config_from_ui(self) -> AppConfig:
        d = self.config.to_dict()
        for k, w in self.inputs.items():
            d[k] = w.text().strip()
        for k, w in self.params.items():
            raw = w.currentText() if isinstance(w, QComboBox) else w.text().strip()
            current_val = getattr(self.config, k)
            if isinstance(current_val, bool):
                d[k] = raw.lower() == "true"
            elif isinstance(current_val, int):
                d[k] = int(float(raw))
            elif isinstance(current_val, float):
                d[k] = float(raw)
            else:
                d[k] = raw
        return AppConfig.from_dict(d)

    def _confirm_overwrite_later_steps(self) -> bool:
        if not self.pipeline_state:
            return True
        if not (self.pipeline_state.get("step2") or self.pipeline_state.get("step3")):
            return True
        reply = QMessageBox.question(
            self,
            "Overwrite results",
            "This will discard Step 2/3 results in memory. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _ensure_input_files(self, keys) -> bool:
        try:
            self.config = self._read_config_from_ui()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid parameters", str(e))
            return False
        missing = []
        for key in keys:
            if key == "dem_paths":
                try:
                    raws = self.config.parse_dem_paths()
                except ValueError as e:
                    QMessageBox.warning(self, "Invalid parameters", str(e))
                    return False
                missing.extend(str(self.config.resolve_path(raw)) for raw in raws if not self.config.resolve_path(raw).exists())
            else:
                path = self.config.resolve_path(getattr(self.config, key))
                if not path.exists():
                    missing.append(str(path))
        if not missing:
            return True
        msg = (
            "Missing input files. Put the datasets under the program data/ folder "
            "(see data/README.md):\n\n" + "\n".join(missing)
        )
        self.append_log(msg)
        QMessageBox.critical(self, "Missing input files", msg)
        return False

    def _apply_vis_to_step3_params(self) -> None:
        epoch_idx = self.cmb_epoch.currentData()
        if epoch_idx is not None and int(epoch_idx) >= 0:
            self.config.day_idx = int(epoch_idx)
            w = self.params.get("day_idx")
            if w is not None:
                w.setText(str(self.config.day_idx))
        dir_map = {"N": 0, "E": 1, "Z": 2}
        if self.cmb_nez_dir.currentText() in dir_map:
            self.config.dir_idx = dir_map[self.cmb_nez_dir.currentText()]
            w = self.params.get("dir_idx")
            if w is not None:
                w.setText(str(self.config.dir_idx))

    def _apply_step3_params_to_vis(self) -> None:
        labels = ["N", "E", "Z"]
        dir_idx = int(np.clip(int(self.config.dir_idx), 0, 2))
        self.cmb_nez_dir.blockSignals(True)
        self.cmb_nez_dir.setCurrentText(labels[dir_idx])
        self.cmb_nez_dir.blockSignals(False)
        if self.cmb_epoch.isEnabled() and self.cmb_epoch.count() > 0 and self.cmb_epoch.itemData(0) != -1:
            idx = int(np.clip(int(self.config.day_idx), 0, self.cmb_epoch.count() - 1))
            self.cmb_epoch.blockSignals(True)
            self.cmb_epoch.setCurrentIndex(idx)
            self.cmb_epoch.blockSignals(False)

    def _set_run_enabled(self, enabled: bool):
        for b in [
            self.btn_step11,
            self.btn_step12,
            self.btn_step13,
            self.btn_step1,
            self.btn_step21,
            self.btn_step22,
            self.btn_step23,
            self.btn_step2,
            self.btn_step31,
            self.btn_step32,
            self.btn_step33,
            self.btn_step3,
            self.btn_all,
        ]:
            b.setEnabled(enabled)

    def _start_worker(self, task_name: str, fn, state: Optional[Dict]):
        self.current_task = task_name
        self._set_run_enabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.thread = QThread(self)
        self.worker = PipelineWorker(self.config, step_fn=fn, state=state)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(int(pct))
        self.append_log(f"[{pct:>3}%] {msg}")

    def on_map_control_changed(self):
        try:
            self.config = self._read_config_from_ui()
        except ValueError:
            return
        sender = self.sender()
        if sender in (self.cmb_epoch, self.cmb_nez_dir):
            self._apply_vis_to_step3_params()
        else:
            self._apply_step3_params_to_vis()
        self.render_plots()

    def on_clear_log(self):
        self.log.clear()

    def save_workflow_state(self):
        if not self.pipeline_state:
            QMessageBox.information(self, "Info", "No workflow state to save.")
            return
        try:
            self.config = self._read_config_from_ui()
        except ValueError:
            QMessageBox.warning(self, "Save status", "Please fix invalid parameter values before saving.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save workflow state", "", "State Files (*.pkl)")
        if not out:
            return
        payload = {
            "config": self.config.to_dict(),
            "pipeline_state": self.pipeline_state,
            "selected_polygon_override": self.selected_polygon_override,
        }
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        self.append_log(f"Workflow state saved: {out}")

    def load_workflow_state(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load workflow state", "", "State Files (*.pkl)")
        if not p:
            return
        with open(p, "rb") as f:
            payload = pickle.load(f)
        self.config = AppConfig.from_dict(payload.get("config", {}))
        self.pipeline_state = payload.get("pipeline_state", None)
        self.selected_polygon_override = payload.get("selected_polygon_override", None)
        if self.pipeline_state is not None and self.selected_polygon_override is not None:
            self.pipeline_state["selected_polygon_override"] = self.selected_polygon_override
        self._overlay_cache = {"pipe": None, "fault": None}
        self._fill_from_config(self.config)
        self._update_epoch_selector()
        self._apply_step3_params_to_vis()
        has_state = bool(self.pipeline_state)
        self.btn_save_status.setEnabled(has_state)
        self.btn_export.setEnabled(has_state)
        self.btn_open.setEnabled(True)
        self.render_plots()
        self.append_log(f"Workflow state loaded: {p}")

    def on_run_step11(self):
        if not self._confirm_overwrite_later_steps():
            return
        if not self._ensure_input_files(("dem_paths",)):
            return
        self.append_log("Step1-1 started...")
        self.selected_polygon_override = None
        self._start_worker("step1-1", run_step1_1, {})

    def on_run_step12(self):
        if not self.pipeline_state or self.pipeline_state.get("step1") is None:
            QMessageBox.information(self, "Info", "Please run Step1-1 first.")
            return
        self.config = self._read_config_from_ui()
        self.append_log("Step1-2 started...")
        self._start_worker("step1-2", run_step1_2, self.pipeline_state)

    def on_run_step13(self):
        if not self.pipeline_state or self.pipeline_state.get("step1") is None or self.pipeline_state["step1"].aspect_raw is None:
            QMessageBox.information(self, "Info", "Please run Step1-2 first.")
            return
        self.config = self._read_config_from_ui()
        self.append_log("Step1-3 started...")
        self._start_worker("step1-3", run_step1_3, self.pipeline_state)

    def on_run_step1(self):
        if not self._confirm_overwrite_later_steps():
            return
        if not self._ensure_input_files(("dem_paths",)):
            return
        self.append_log("Step1 (All) started...")
        self.selected_polygon_override = None
        self._start_worker("step1", run_step1, {})

    def on_run_step2(self):
        if not self.pipeline_state or self.pipeline_state.get("step1") is None or self.pipeline_state["step1"].aspect_raw is None:
            QMessageBox.information(self, "Info", "Please run Step1-2 first.")
            return
        if not self._ensure_input_files(("asc_shp_path", "desc_shp_path")):
            return
        self.append_log("Step2 (all) started...")
        self._start_worker("step2", run_step2, self.pipeline_state)

    def on_run_step21(self):
        if not self.pipeline_state or self.pipeline_state.get("step1") is None or self.pipeline_state["step1"].aspect_raw is None:
            QMessageBox.information(self, "Info", "Please run Step1-2 first.")
            return
        if not self._ensure_input_files(("asc_shp_path", "desc_shp_path")):
            return
        self._overlay_cache = {"pipe": None, "fault": None}
        self.append_log("Step2-1 started...")
        self._start_worker("step2-1", run_step2_1, self.pipeline_state)

    def on_run_step22(self):
        if not self.pipeline_state or self.pipeline_state.get("step1") is None or self.pipeline_state["step1"].insar_aspect is None:
            QMessageBox.information(self, "Info", "Please run Step2-1 first.")
            return
        self.config = self._read_config_from_ui()
        self.append_log("Step2-2 started...")
        self._start_worker("step2-2", run_step2_2, self.pipeline_state)

    def on_run_step23(self):
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            QMessageBox.information(self, "Info", "Please run Step2-2 first.")
            return
        self.config = self._read_config_from_ui()
        self.append_log("Step2-3 started...")
        self._start_worker("step2-3", run_step2_3, self.pipeline_state)

    def _sync_polygon_into_state(self):
        if self.pipeline_state is None:
            return
        self.pipeline_state["selected_polygon_override"] = self.selected_polygon_override

    def on_run_step3(self):
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            QMessageBox.information(self, "Info", "Please run Step2 first.")
            return
        self.config = self._read_config_from_ui()
        if not self.on_select_contour(silent=True):
            self.append_log("Step3 cancelled: contour not selected.")
            return
        self._sync_polygon_into_state()
        self.append_log("Step3 (pick + mesh + strain) started...")
        self._start_worker("step3", run_step3, self.pipeline_state)

    def on_run_step31(self):
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            QMessageBox.information(self, "Info", "Please run Step2 first.")
            return
        self.config = self._read_config_from_ui()
        if not self.on_select_contour(silent=False):
            return
        self._sync_polygon_into_state()
        self.append_log("Step3-1 pick commit started...")
        self._start_worker("step3-pick", run_step3_pick, self.pipeline_state)

    def on_run_step32(self):
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            QMessageBox.information(self, "Info", "Please run Step2 first.")
            return
        if not self.pipeline_state.get("step3_selected_lines_path"):
            QMessageBox.information(self, "Info", "Please run Step3-1 Pick Contour first.")
            return
        self.config = self._read_config_from_ui()
        self.append_log("Step3-2 mesh started...")
        self._start_worker("step3-mesh", run_step3_mesh, self.pipeline_state)

    def on_run_step33(self):
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            QMessageBox.information(self, "Info", "Please run Step2 first.")
            return
        if self.pipeline_state.get("step3_mesh") is None:
            QMessageBox.information(self, "Info", "Please run Step3-2 Mesh Pipeline first.")
            return
        self.config = self._read_config_from_ui()
        self._sync_polygon_into_state()
        self.append_log("Step3-3 mapping + strain started...")
        self._start_worker("step3-response", run_step3_response, self.pipeline_state)

    def on_run_all(self):
        if not self._confirm_overwrite_later_steps():
            return
        if not self._ensure_input_files(("dem_paths", "asc_shp_path", "desc_shp_path")):
            return
        self.selected_polygon_override = None
        self.append_log("Running Step1 and Step2; contour pick will follow...")
        self._start_worker("all-pre", run_steps_1_and_2, {})

    def on_select_contour(self, silent: bool = False) -> bool:
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            if not silent:
                QMessageBox.information(self, "Info", "Please run Step2 first.")
            return False
        self.config = self._read_config_from_ui()
        try:
            ok = run_contour_picker(
                self.config,
                self.pipeline_state,
                log=self.append_log,
                overlay_cache=self._overlay_cache,
                parent=self,
            )
            if ok:
                self.selected_polygon_override = self.pipeline_state.get("selected_polygon_override")
                if not silent:
                    QMessageBox.information(self, "Contour Saved", "Contour GIS saved. Continue with Step3-2 Mesh.")
            return ok
        except Exception as e:
            self.append_log(f"Contour selection failed: {e}")
            if not silent:
                QMessageBox.critical(self, "Contour selection failed", str(e))
            return False

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.append_log("Stop signal sent.")

    def on_finished(self, result: Dict):
        task = self.current_task
        self.pipeline_state = result
        if result.get("selected_polygon_override") is not None:
            self.selected_polygon_override = result.get("selected_polygon_override")
        continue_all = task == "all-pre" and bool(result.get("needs_contour_pick"))
        self._set_run_enabled(not continue_all)
        self.btn_stop.setEnabled(False)
        self.btn_save_status.setEnabled(bool(self.pipeline_state))
        self.btn_export.setEnabled(bool(self.pipeline_state))
        self.btn_open.setEnabled(True)
        self._update_epoch_selector()
        self._apply_step3_params_to_vis()
        self.progress.setValue(100)
        self.append_log(f"{task} completed.")
        self.render_plots()
        if continue_all:
            # Let the Step1/2 worker thread finish before opening the modal picker
            # and starting Step3. Starting a new QThread from this slot is unsafe.
            QTimer.singleShot(0, self._continue_run_all_after_contour)

    def _continue_run_all_after_contour(self):
        if not self.pipeline_state or self.pipeline_state.get("step2") is None:
            self._set_run_enabled(True)
            self.append_log("Run All stopped after Step2: Step2 results missing.")
            return
        self.append_log("Run All: please pick a contour to continue Step3.")
        if not self.on_select_contour(silent=True):
            self._set_run_enabled(True)
            self.append_log("Run All stopped after Step2: contour not selected.")
            return
        try:
            self.config = self._read_config_from_ui()
        except ValueError:
            self._set_run_enabled(True)
            self.append_log("Run All stopped: invalid Step3 parameters.")
            return
        self._sync_polygon_into_state()
        self.append_log("Run All: contour picked; running Step 3-2 mesh and Step 3-3 strain...")
        self._start_worker("step3", run_step3, self.pipeline_state)

    def on_failed(self, tb: str):
        self._set_run_enabled(True)
        self.btn_stop.setEnabled(False)
        text = tb or ""
        if text.startswith("CANCELLED") or "InterruptedError" in text:
            self.append_log("Computation stopped by user.")
            QMessageBox.information(self, "Stopped", "Computation stopped.")
            return
        self.append_log("Run failed:")
        self.append_log(text)
        QMessageBox.critical(self, "Run Failed", text.splitlines()[-1] if text else "Unknown error")

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _parse_optional_float(self, text: str):
        t = str(text).strip()
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            return None

    def _update_epoch_selector(self):
        self.cmb_epoch.blockSignals(True)
        self.cmb_epoch.clear()
        step1 = self.pipeline_state.get("step1") if self.pipeline_state else None
        if step1 is None or not step1.matched_dates:
            self.cmb_epoch.addItem("Run Step2 to load epochs", -1)
            self.cmb_epoch.setEnabled(False)
            self.cmb_epoch.blockSignals(False)
            return
        for i, pair in enumerate(step1.matched_dates):
            self.cmb_epoch.addItem(f"{i}: {pair[0]} / {pair[1]}", i)
        self.cmb_epoch.setCurrentIndex(min(max(int(self.config.day_idx), 0), self.cmb_epoch.count() - 1))
        self.cmb_epoch.setEnabled(True)
        self.cmb_epoch.blockSignals(False)

    def _clone_scalar_mappable(self, src, dst) -> None:
        arr = src.get_array() if hasattr(src, "get_array") else None
        if arr is None or np.size(arr) == 0:
            return
        dst.set_array(np.asarray(arr, dtype=float))
        try:
            dst.set_cmap(src.get_cmap())
        except Exception:
            pass
        try:
            dst.set_norm(src.norm)
        except Exception:
            pass
        try:
            dst.set_clim(*src.get_clim())
        except Exception:
            pass

    def _clone_collection(self, coll, ax):
        alpha = coll.get_alpha()
        zorder = coll.get_zorder()
        if isinstance(coll, LineCollection):
            new = LineCollection(coll.get_segments())
            new.set_linewidths(coll.get_linewidths())
            new.set_linestyles(coll.get_linestyles())
            new.set_zorder(zorder)
            if alpha is not None:
                new.set_alpha(alpha)
            arr = coll.get_array()
            if arr is not None and np.size(arr) > 0:
                self._clone_scalar_mappable(coll, new)
            else:
                new.set_edgecolor(coll.get_edgecolor())
                fc = coll.get_facecolor()
                if fc is not None and len(np.asarray(fc)):
                    new.set_facecolor(fc)
            ax.add_collection(new)
            return new if (arr is not None and np.size(arr) > 0) else None
        if isinstance(coll, PathCollection):
            offs = np.asarray(coll.get_offsets(), dtype=float)
            if offs.ndim != 2 or offs.shape[0] == 0:
                return None
            sizes = np.asarray(coll.get_sizes(), dtype=float)
            s = sizes if sizes.size else 2.0
            arr = coll.get_array()
            kwargs = {
                "s": s,
                "linewidths": coll.get_linewidths(),
                "edgecolors": coll.get_edgecolors(),
                "alpha": 1.0 if alpha is None else alpha,
                "zorder": zorder,
            }
            if arr is not None and np.size(arr) > 0:
                try:
                    kwargs["cmap"] = coll.get_cmap()
                    kwargs["norm"] = coll.norm
                except Exception:
                    pass
                return ax.scatter(offs[:, 0], offs[:, 1], c=np.asarray(arr), **kwargs)
            return ax.scatter(offs[:, 0], offs[:, 1], c=coll.get_facecolors(), **kwargs)
        if isinstance(coll, PolyCollection) or hasattr(coll, "get_paths"):
            try:
                verts = [np.asarray(p.vertices) for p in coll.get_paths()]
            except Exception:
                return None
            if not verts:
                return None
            new = PolyCollection(verts)
            new.set_facecolor(coll.get_facecolors())
            new.set_edgecolor(coll.get_edgecolors())
            new.set_linewidths(coll.get_linewidths())
            new.set_zorder(zorder)
            if alpha is not None:
                new.set_alpha(alpha)
            arr = coll.get_array()
            if arr is not None and np.size(arr) > 0:
                self._clone_scalar_mappable(coll, new)
            ax.add_collection(new)
            return new if (arr is not None and np.size(arr) > 0) else None
        return None

    def _rebuild_axes_from(self, src_ax, src_fig) -> Figure:
        fig = Figure(figsize=(8, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        last_mappable = None
        for im in list(src_ax.images):
            kwargs = {
                "origin": im.origin,
                "extent": im.get_extent(),
                "interpolation": im.get_interpolation(),
                "alpha": 1.0 if im.get_alpha() is None else im.get_alpha(),
                "zorder": im.get_zorder(),
                "aspect": "auto",
            }
            try:
                kwargs["cmap"] = im.get_cmap()
            except Exception:
                pass
            try:
                kwargs["norm"] = im.norm
            except Exception:
                pass
            mapped = ax.imshow(np.asarray(im.get_array()), **kwargs)
            arr = mapped.get_array()
            if arr is not None and np.size(arr) > 0 and len(src_ax.collections) == 0:
                last_mappable = mapped
        for coll in list(src_ax.collections):
            mapped = self._clone_collection(coll, ax)
            if mapped is not None:
                last_mappable = mapped
        for line in list(src_ax.lines):
            x, y = line.get_data()
            ax.plot(
                x,
                y,
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                alpha=1.0 if line.get_alpha() is None else line.get_alpha(),
                zorder=line.get_zorder(),
            )
        for txt in list(src_ax.texts):
            x, y = txt.get_position()
            kw = {
                "fontsize": txt.get_fontsize(),
                "color": txt.get_color(),
                "ha": txt.get_ha(),
                "va": txt.get_va(),
                "zorder": txt.get_zorder(),
                "alpha": 1.0 if txt.get_alpha() is None else txt.get_alpha(),
            }
            try:
                if txt.get_transform() is src_ax.transAxes:
                    kw["transform"] = ax.transAxes
            except Exception:
                pass
            ax.text(x, y, txt.get_text(), **kw)
        leg = src_ax.get_legend()
        if leg is not None:
            src_handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
            handles, labels = [], []
            for h, t in zip(src_handles, leg.get_texts()):
                color = "k"
                lw = 3.5
                try:
                    color = h.get_color()
                except Exception:
                    try:
                        color = h.get_edgecolor()
                    except Exception:
                        pass
                try:
                    lw = h.get_linewidth()
                except Exception:
                    pass
                handles.append(Line2D([0], [0], color=color, lw=lw))
                labels.append(t.get_text())
            title = ""
            try:
                title = leg.get_title().get_text()
            except Exception:
                title = ""
            if handles:
                ax.legend(
                    handles=handles,
                    labels=labels,
                    loc="upper left",
                    framealpha=0.9,
                    title=title or None,
                )
        ax.set_title(src_ax.get_title())
        ax.set_xlabel(src_ax.get_xlabel())
        ax.set_ylabel(src_ax.get_ylabel())
        ax.set_xlim(src_ax.get_xlim())
        ax.set_ylim(src_ax.get_ylim())
        aspect = src_ax.get_aspect()
        if aspect == "equal":
            ax.set_aspect("equal", adjustable="box")
        else:
            try:
                ax.set_aspect(aspect)
            except Exception:
                ax.set_aspect("auto")
        if last_mappable is not None and len(src_fig.axes) > 1:
            cbar = fig.colorbar(last_mappable, ax=ax, shrink=0.8)
            for other in src_fig.axes:
                if other is src_ax:
                    continue
                label = other.get_ylabel()
                if label:
                    cbar.set_label(label)
                break
        return fig

    def on_open_map_popup(self):
        idx = self.tabs.currentIndex()
        plots = getattr(self, "_vis_plot_tabs", None) or [self.tab_insar]
        src_tab = plots[idx] if 0 <= idx < len(plots) else self.tab_insar
        fig = self._rebuild_axes_from(src_tab.ax, src_tab.figure)
        w = QMainWindow(self)
        w.setWindowTitle(self.tabs.tabText(idx))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, w)
        cw = QWidget()
        lay = QVBoxLayout(cw)
        lay.addWidget(toolbar)
        lay.addWidget(canvas)
        w.setCentralWidget(cw)
        w.resize(980, 700)
        w.show()
        w._hold = (canvas, toolbar, fig)

    def render_plots(self):
        render_plots(self)

    def _collect_figures(self) -> Dict[str, object]:
        return {
            "dem_aspect": self.tab_dem.figure,
            "insar_map": self.tab_insar.figure,
            "mesh": self.tab_mesh.figure,
            "pipeline_age": self.tab_pipe_age.figure,
            "pipeline_material": self.tab_pipe_mat.figure,
            "pipeline_diameters": self.tab_pipe_diam.figure,
            "pipeline_displacement": self.tab_disp.figure,
            "pipeline_strain": self.tab_strain.figure,
        }

    def on_export(self):
        if not self.pipeline_state:
            QMessageBox.information(self, "Info", "No results to export.")
            return
        try:
            self.config = self._read_config_from_ui()
            out = export_state(self.config, self.pipeline_state, figures=self._collect_figures())
            self.append_log(f"Exported run bundle: {out}")
            QMessageBox.information(self, "Export", f"Export completed:\n{out}")
        except ValueError:
            QMessageBox.warning(self, "Export", "Please fix invalid parameter values before exporting.")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def on_open_output(self):
        try:
            self.config = self._read_config_from_ui()
        except ValueError:
            QMessageBox.warning(self, "Open folder", "Please fix invalid parameter values first.")
            return
        folder = Path(self.config.resolve_path(self.config.output_dir))
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def reset_defaults(self):
        self.config = AppConfig()
        self.selected_polygon_override = None
        self._fill_from_config(self.config)
        self.append_log("Parameters reset to default values.")

    def save_preset(self):
        self.config = self._read_config_from_ui()
        out, _ = QFileDialog.getSaveFileName(self, "Save parameter preset", "", "JSON Files (*.json)")
        if not out:
            return
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        self.append_log(f"Preset saved: {out}")

    def load_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load parameter preset", "", "JSON Files (*.json)")
        if not p:
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.config = AppConfig.from_dict(data)
        self.selected_polygon_override = None
        self._fill_from_config(self.config)
        self.append_log(f"Preset loaded: {p}")
