# -*- coding: utf-8 -*-
"""
EEG labeling app + read-only fusion model suggestion.

Behavior:
- Loads a 1D .npy EEG signal
- Displays EEG, spectrogram, editable mask, and model-predicted mask
- Only shows model bundles whose folder name contains 'fusion'
- Uses BOTH the raw 1D signal window and the computed spectrogram window as model input
- Resamples model predictions to len(t_spec) for display when needed
- Autosaves editable masks to JSON

Expected bundle behavior:
- Bundle loader exists at: DL.src.io.bundle.load_bundle
- Saved fusion models output logits shaped (B,C,T), (B,T,C), or a dict containing 'logits'
- stats contain mean_1d/std_1d and mean_2d/std_2d
"""

from __future__ import annotations

import sys
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, NamedTuple

import numpy as np
import scipy as sc

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QGraphicsEllipseItem, QRubberBand
from PyQt6.QtCore import QRect, QSize, QPointF
import pyqtgraph as pg

import torch
import importlib
import pkgutil

from pathlib import Path
import json
import torch

from DL.src.models.registry import build_fusion

from time import time


# ---------------------------
# CONFIG
# ---------------------------
@dataclass
class AppConfig:
    fs_default: float = 128.0
    window_size_s_default: float = 30.0

    # Folder containing saved bundles
    runs_dir: str = "DL/runs"

    # Only bundle folders containing this token will be shown
    model_name_token: str = "fusion"

    default_bundle: Optional[str] = None
    device: str = "cuda"

    # Spectrogram settings for display + 2D branch inference
    f_cut_hz: float = 45.0
    nperseg_factor: float = 1.0
    noverlap_factor: float = 0.90
    nfft_factor: float = 1.0


CFG = AppConfig()


# ---------------------------
# Spectrogram (display + inference 2D branch)
# ---------------------------
def spectrogram(
    y,
    fs,
    nperseg_factor=1,
    noverlap_factor=0.9,
    nfft_factor=1,
    detrend=False,
    scaling="psd",
    f_cut=45,
):
    nperseg = int(nperseg_factor * fs)
    nperseg = max(nperseg, 2)
    noverlap = int(noverlap_factor * nperseg)
    noverlap = min(noverlap, nperseg - 1)
    nfft = int(nfft_factor * nperseg)
    nfft = max(nfft, nperseg)

    window = sc.signal.windows.hamming(nperseg, sym=True)

    f_spectro, t_spectro, stft = sc.signal.stft(
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
    )

    Sxx = np.abs(stft) ** 2

    if len(f_spectro) > 1:
        df = f_spectro[1] - f_spectro[0]
        j = int(f_cut / df)
        j = max(1, min(j, len(f_spectro)))
        f_spectro = f_spectro[:j]
        Sxx = Sxx[:j, :]

    return f_spectro, t_spectro, Sxx


def make_jet_lut(n: int = 256) -> np.ndarray:
    try:
        from matplotlib import cm
        cmap = cm.get_cmap("jet", n)
        lut = (cmap(np.linspace(0, 1, n))[:, :4] * 255).astype(np.ubyte)
        return lut
    except Exception:
        pass

    try:
        cm_pg = pg.colormap.get("jet")
        lut = (cm_pg.getLookupTable(0.0, 1.0, n, alpha=True) * 255).astype(np.ubyte)
        return lut
    except Exception:
        pass

    x = np.linspace(0, 1, n)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    a = np.ones_like(x)
    return (np.stack([r, g, b, a], axis=1) * 255).astype(np.ubyte)


def make_class_lut(colors_rgba: dict, vmax: int, n: int = 256) -> np.ndarray:
    lut = np.zeros((n, 4), dtype=np.ubyte)
    for i in range(n):
        k = int(np.clip(round(i * vmax / (n - 1)), 0, vmax))
        lut[i] = np.array(colors_rgba.get(k, (180, 180, 180, 220)), dtype=np.ubyte)
    return lut


# ---------------------------
# Mask classes
# ---------------------------
MASK_CLASSES = [
    "ok",
    "alpha-sup",
    "IES",
    "gc",
    "shallow",
    "gamma",
    "eye artifact",
    "HF artifact",
    "large artifact",
    "awake",
]

MASK_COLORS_RGBA = {
    0: (80, 200, 120, 220),
    1: (120, 120, 255, 220),
    2: (255, 170, 0, 220),
    3: (30, 30, 30, 220),
    4: (200, 200, 0, 220),
    5: (255, 0, 200, 220),
    6: (0, 220, 220, 220),
    7: (255, 0, 0, 220),
    8: (150, 75, 0, 220),
    9: (100, 75, 0, 220),
}
MASK_MAX = max(MASK_COLORS_RGBA.keys())


# ---------------------------
# Utility functions
# ---------------------------
def _resample_labels_nearest(labels: np.ndarray, out_T: int) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {labels.shape}")
    if len(labels) == out_T:
        return labels.astype(int)

    x_old = np.linspace(0.0, 1.0, len(labels))
    x_new = np.linspace(0.0, 1.0, out_T)
    idx = np.clip(
        np.round(np.interp(x_new, x_old, np.arange(len(labels)))).astype(int),
        0,
        len(labels) - 1,
    )
    return labels[idx].astype(int)


def _import_all_src_models():
    try:
        import DL.src.models
        pkg = DL.src.models
        for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(m.name)
    except Exception:
        pass


_import_all_src_models()


# ---------------------------
# Mask plot helpers
# ---------------------------
class RubberbandMaskPlot(pg.PlotWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.origin = None
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        modifiers = event.modifiers()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and (modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier):
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()
            self.plotItem.vb.setMouseEnabled(x=False, y=False)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubberBand.isVisible() and self.origin is not None:
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubberBand.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        modifiers = event.modifiers()
        ctrl_held = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift_held = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

        if self.rubberBand.isVisible():
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()

            top_left = self.plotItem.vb.mapSceneToView(self.mapToScene(rect.topLeft()))
            bottom_right = self.plotItem.vb.mapSceneToView(self.mapToScene(rect.bottomRight()))
            xmin, xmax = sorted([top_left.x(), bottom_right.x()])
            ymin, ymax = sorted([top_left.y(), bottom_right.y()])

            for p in self.viewer.mask_points:
                pos = p.pos()
                if xmin <= pos.x() <= xmax and ymin <= pos.y() <= ymax:
                    if ctrl_held:
                        p.set_selected(False)
                    elif shift_held:
                        p.set_selected(True)

            self.plotItem.vb.setMouseEnabled(x=True, y=True)
            self.viewer._update_status_line()
            return

        super().mouseReleaseEvent(event)


class DraggableMaskPoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=5, index=None, viewer=None):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setPos(x, y)
        self.setPen(pg.mkPen("k"))
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsScenePositionChanges, True)

        self.index = index
        self.fixed_x = float(x)
        self.viewer = viewer
        self.selected = False
        self.set_selected(False)

    def set_selected(self, selected: bool):
        self.selected = bool(selected)
        val = int(round(self.pos().y()))
        val = int(np.clip(val, 0, MASK_MAX))
        color = MASK_COLORS_RGBA.get(val, (180, 180, 180, 220))
        self.setBrush(pg.mkBrush(*color))
        self.setPen(pg.mkPen("w" if self.selected else "k", width=3 if self.selected else 1))

    def mousePressEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.set_selected(not self.selected)
            if self.viewer is not None:
                self.viewer._update_status_line()
            event.accept()
            return
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionChange:
            new_y = int(round(float(value.y())))
            new_y = int(np.clip(new_y, 0, MASK_MAX))
            new_pos = QPointF(self.fixed_x, float(new_y))

            if self.viewer is not None:
                if self.selected:
                    self.viewer._group_set_selected_mask(new_y)
                else:
                    self.viewer._set_mask_value(self.index, new_y)

            return new_pos
        return super().itemChange(change, value)


# ---------------------------
# EEG rubberband selection / label
# ---------------------------
class EEGLabelRubberBandFilter(QtCore.QObject):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.origin = None
        self.rb = QRubberBand(QRubberBand.Shape.Rectangle, viewer.plot_signal.viewport())

    def eventFilter(self, obj, event):
        v = self.viewer
        if obj is not v.plot_signal.viewport() or not (v._rb_zoom_active or v._rb_label_active):
            return False

        if event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            self.rb.setGeometry(QRect(self.origin, QSize()))
            self.rb.show()
            return True

        if event.type() == QtCore.QEvent.Type.MouseMove and self.origin is not None:
            rect = QRect(self.origin, event.position().toPoint()).normalized()
            self.rb.setGeometry(rect)
            return True

        if event.type() == QtCore.QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.origin is None:
                return True

            rect = self.rb.geometry().normalized()
            self.rb.hide()
            self.origin = None

            if rect.width() < 4 or rect.height() < 4:
                return True

            scene_tl = v.plot_signal.mapToScene(rect.topLeft())
            scene_br = v.plot_signal.mapToScene(rect.bottomRight())
            vb = v.plot_signal.plotItem.vb
            data_tl = vb.mapSceneToView(scene_tl)
            data_br = vb.mapSceneToView(scene_br)
            x0, x1 = sorted([float(data_tl.x()), float(data_br.x())])

            if v._rb_label_active:
                v._apply_eeg_label_to_mask(x0, x1)
            else:
                v.plot_signal.setXRange(x0, x1, padding=0)
                v.plot_spec.setXRange(x0, x1, padding=0)
                v.plot_mask.setXRange(x0, x1, padding=0)
                v.plot_model_mask.setXRange(x0, x1, padding=0)
                v.select_btn.setChecked(False)

            return True

        return False


# ---------------------------
# Bundle predictor for fusion models only
# ---------------------------
class ModelBundle(NamedTuple):
    model: "torch.nn.Module"
    mean_1d: float
    std_1d: float
    mean_2d: float
    std_2d: float
    num_classes: int
    arch: str
    bundle_dir: str


def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
        if hasattr(d, k):
            return getattr(d, k)
    return default


class DLBundledPredictorFusion:
    def __init__(self, bundle_dir: str, device_pref: str = "cuda"):
        self.bundle_dir = bundle_dir
        self.device_pref = device_pref

        dev = "cpu"
        if device_pref.lower().startswith("cuda") and torch.cuda.is_available():
            dev = "cuda"
        self._device = torch.device(dev)

        self.bundle: Optional[ModelBundle] = None
        self._load()

    @property
    def device(self):
        return self._device

    def _load(self):
        bundle_dir = Path(self.bundle_dir)

        # -------------------------
        # Locate bundle files
        # -------------------------
        config_path = bundle_dir / "config.json"
        stats_path = bundle_dir / "stats.json"

        # weights: support common names
        candidate_weight_files = [
            bundle_dir / "model.pt",
            bundle_dir / "best_model.pt",
            bundle_dir / "weights.pt",
        ]
        weights_path = None
        for p in candidate_weight_files:
            if p.exists():
                weights_path = p
                break

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing stats file: {stats_path}")
        if weights_path is None:
            raise FileNotFoundError(
                f"Could not find model weights in {bundle_dir}. "
                f"Tried: {[str(p.name) for p in candidate_weight_files]}"
            )

        # -------------------------
        # Read config + stats
        # -------------------------
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = json.load(f)

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # The training script saved model_cfg=cfg["model"]
        # so config.json may already BE the model config,
        # or it may contain {"model": ...}. Support both.
        if isinstance(full_cfg, dict) and "name" in full_cfg and "kwargs" in full_cfg:
            model_cfg = full_cfg
        elif isinstance(full_cfg, dict) and "model" in full_cfg:
            model_cfg = full_cfg["model"]
        else:
            raise ValueError(
                f"Could not infer fusion model config format from {config_path}"
            )

        # -------------------------
        # Rebuild fusion architecture the same way as training
        # -------------------------
        model = build_fusion(model_cfg)
        model.to(self._device)

        # -------------------------
        # Load weights
        # -------------------------
        state = torch.load(weights_path, map_location=self._device)

        # Some save formats store {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)

        model.eval()

        # -------------------------
        # Stats
        # -------------------------
        mean_1d = float(stats.get("mean_1d", 0.0))
        std_1d  = float(stats.get("std_1d", 1.0))
        mean_2d = float(stats.get("mean_2d", 0.0))
        std_2d  = float(stats.get("std_2d", 1.0))

        std_1d = max(abs(std_1d), 1e-8)
        std_2d = max(abs(std_2d), 1e-8)

        num_classes = int(model_cfg.get("kwargs", {}).get("num_classes", 10))
        arch = str(model_cfg.get("name", "unknown"))

        self.bundle = ModelBundle(
            model=model,
            mean_1d=mean_1d,
            std_1d=std_1d,
            mean_2d=mean_2d,
            std_2d=std_2d,
            num_classes=num_classes,
            arch=arch,
            bundle_dir=str(bundle_dir),
        )

        if missing:
            print("Warning - missing keys when loading fusion model:", missing)
        if unexpected:
            print("Warning - unexpected keys when loading fusion model:", unexpected)

    def predict_mask_from_signal_and_sxx(self, sig: np.ndarray, sxx: np.ndarray, out_T: int) -> np.ndarray:
        if self.bundle is None:
            raise RuntimeError("Bundle not loaded")

        if sig.ndim != 1:
            raise ValueError(f"Expected 1D signal, got shape {sig.shape}")
        if sxx.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape {sxx.shape}")

        x1d = sig.astype(np.float32)
        x1d = (x1d - self.bundle.mean_1d) / self.bundle.std_1d
        x1d_t = torch.from_numpy(x1d).float().unsqueeze(0).unsqueeze(0).to(self._device)

        x2d = np.log1p(sxx.astype(np.float32) + 1e-11)
        x2d = (x2d - self.bundle.mean_2d) / self.bundle.std_2d
        x2d_t = torch.from_numpy(x2d).float().unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            y = self.bundle.model(x1d_t, x2d_t)

        if isinstance(y, dict):
            logits = y.get("logits", None)
            if logits is None:
                raise ValueError("Fusion model returned a dict without 'logits'")
        elif isinstance(y, (tuple, list)):
            logits = y[0]
        else:
            logits = y

        if logits.ndim != 3:
            raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")

        if logits.shape[1] == self.bundle.num_classes:
            pass
        elif logits.shape[2] == self.bundle.num_classes:
            logits = logits.permute(0, 2, 1)
        else:
            raise ValueError(
                f"Cannot infer class dimension from output shape {tuple(logits.shape)} "
                f"with num_classes={self.bundle.num_classes}"
            )

        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(int)
        pred = _resample_labels_nearest(pred, out_T=out_T)
        return pred


# ---------------------------
# Main app
# ---------------------------
class EEGLabelerWithModelFusion(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG labeling + fusion model suggestion (read-only)")
        self.resize(950, 1050)

        self.fs: float = CFG.fs_default

        self.eeg: Optional[np.ndarray] = None
        self.path_npy: Optional[str] = None

        self.window_size_s: float = CFG.window_size_s_default
        self.window_idx: int = 0

        self.t_sig: np.ndarray = np.array([])
        self.sig_win: np.ndarray = np.array([])

        self.t_spec: np.ndarray = np.array([])
        self.f: np.ndarray = np.array([])
        self.Sxx: Optional[np.ndarray] = None

        self.mask: np.ndarray = np.array([])
        self._original_mask: Optional[np.ndarray] = None
        self._dirty: bool = False

        self.model_mask: np.ndarray = np.array([])
        self._model_ok: bool = False
        self.predictor: Optional[DLBundledPredictorFusion] = None

        self.mask_points: List[DraggableMaskPoint] = []
        self._updating_group = False

        self.overlay_curve_item = None
        self.threshold_line_item = None

        self._rb_zoom_active = False
        self._rb_label_active = False

        self._jet_lut = make_jet_lut(256)
        self._class_lut = make_class_lut(MASK_COLORS_RGBA, MASK_MAX, 256)

        self._build_ui()
        self._connect_signals()

        self._eeg_rb_filter = EEGLabelRubberBandFilter(self)
        self.plot_signal.viewport().installEventFilter(self._eeg_rb_filter)

        QtGui.QShortcut(QtGui.QKeySequence("Left"), self, activated=self.prev_window)
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self, activated=self.next_window)
        QtGui.QShortcut(QtGui.QKeySequence("Up"), self, activated=lambda: self._nudge_selected(+1))
        QtGui.QShortcut(QtGui.QKeySequence("Down"), self, activated=lambda: self._nudge_selected(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+G"), self, activated=lambda: self.spin_goto.setFocus())

        self._refresh_bundle_combo()
        self._try_load_model(auto=True)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()

        self.btn_load = QtWidgets.QPushButton("Load .npy…")
        controls.addWidget(self.btn_load)

        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("fs (Hz):"))
        self.spin_fs = QtWidgets.QDoubleSpinBox()
        self.spin_fs.setRange(1.0, 20000.0)
        self.spin_fs.setDecimals(2)
        self.spin_fs.setValue(float(CFG.fs_default))
        self.spin_fs.setFixedWidth(110)
        controls.addWidget(self.spin_fs)

        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Window (s):"))
        self.spin_win = QtWidgets.QDoubleSpinBox()
        self.spin_win.setRange(1.0, 3600.0)
        self.spin_win.setDecimals(1)
        self.spin_win.setValue(float(CFG.window_size_s_default))
        self.spin_win.setFixedWidth(110)
        controls.addWidget(self.spin_win)

        self.btn_prev = QtWidgets.QPushButton("← Prev")
        self.btn_next = QtWidgets.QPushButton("Next →")
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_next)

        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Window:"))
        self.lbl_winpos = QtWidgets.QLabel("- / -")
        self.lbl_winpos.setMinimumWidth(70)
        controls.addWidget(self.lbl_winpos)

        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Go to:"))
        self.spin_goto = QtWidgets.QSpinBox()
        self.spin_goto.setKeyboardTracking(False)
        self.spin_goto.setRange(1, 1)
        self.spin_goto.setValue(1)
        self.spin_goto.setFixedWidth(80)
        controls.addWidget(self.spin_goto)

        controls.addSpacing(16)
        self.select_btn = QtWidgets.QPushButton("Select Range")
        self.select_btn.setCheckable(True)
        controls.addWidget(self.select_btn)

        self.label_btn = QtWidgets.QPushButton("Label Mode")
        self.label_btn.setCheckable(True)
        controls.addWidget(self.label_btn)

        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Class:"))
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.addItems([f"{i}: {name}" for i, name in enumerate(MASK_CLASSES)])
        self.class_combo.setFixedWidth(180)
        controls.addWidget(self.class_combo)

        self.home_btn = QtWidgets.QPushButton("Home")
        controls.addWidget(self.home_btn)

        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 50.0)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setSingleStep(1.0)
        self.threshold_spin.setValue(10.0)
        self.threshold_spin.setFixedWidth(90)
        controls.addWidget(self.threshold_spin)

        controls.addStretch(1)
        self.lbl_status = QtWidgets.QLabel("No file loaded.")
        self.lbl_status.setMinimumWidth(280)
        controls.addWidget(self.lbl_status)

        vbox.addLayout(controls)

        controls_2 = QtWidgets.QHBoxLayout()
        self.btn_minus = QtWidgets.QPushButton("Selected -1")
        self.btn_plus = QtWidgets.QPushButton("Selected +1")
        controls_2.addWidget(self.btn_minus)
        controls_2.addWidget(self.btn_plus)

        self.lbl_dirty = QtWidgets.QLabel("")
        controls_2.addWidget(self.lbl_dirty)

        controls_2.addStretch(1)
        tip = QtWidgets.QLabel("Editable mask: Ctrl-click select; Shift-drag box select; drag vertically to set class")
        tip.setStyleSheet("color:#666;")
        controls_2.addWidget(tip)

        controls_2.addSpacing(16)
        controls_2.addWidget(QtWidgets.QLabel("Fusion model bundle:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(260)
        controls_2.addWidget(self.model_combo)

        self.btn_browse_bundle = QtWidgets.QPushButton("Browse…")
        controls_2.addWidget(self.btn_browse_bundle)

        self.btn_reload_model = QtWidgets.QPushButton("Load selected")
        controls_2.addWidget(self.btn_reload_model)

        self.lbl_model = QtWidgets.QLabel("model: (not loaded)")
        self.lbl_model.setStyleSheet("color:#666;")
        controls_2.addWidget(self.lbl_model)

        vbox.addLayout(controls_2)

        self.plot_signal = pg.PlotWidget(title="EEG (current window)")
        self.plot_signal.setLabel("bottom", "Time", units="s")
        self.plot_signal.setLabel("left", "Amplitude")
        self.plot_signal.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_signal, stretch=2)
        self.curve_sig = self.plot_signal.plot([], [], pen=pg.mkPen(width=1))

        self.plot_spec = pg.PlotWidget(title="Spectrogram (display + fusion 2D input)")
        self.plot_spec.setLabel("bottom", "Time", units="s")
        self.plot_spec.setLabel("left", "Frequency", units="Hz")
        self.plot_spec.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_spec, stretch=3)

        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(self._jet_lut)
        self.img_item.setOpts(axisOrder="row-major")
        self.plot_spec.addItem(self.img_item)

        self.overlay_curve_item = pg.PlotDataItem(pen=pg.mkPen(width=2))
        self.plot_spec.addItem(self.overlay_curve_item)
        self.threshold_line_item = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_spec.addItem(self.threshold_line_item)

        self.plot_mask = RubberbandMaskPlot(self)
        self.plot_mask.setTitle("Mask (editable) — class per spectrogram time-bin")
        self.plot_mask.setLabel("bottom", "Time", units="s")
        self.plot_mask.setLabel("left", "Class")
        self.plot_mask.showGrid(x=True, y=True, alpha=0.3)
        self.plot_mask.setYRange(-0.5, MASK_MAX + 0.5)
        vbox.addWidget(self.plot_mask, stretch=2)

        self.mask_curve = self.plot_mask.plot([], [], pen=pg.mkPen(width=1))

        legend = self.plot_mask.addLegend(offset=(10, 10))
        for v, name in enumerate(MASK_CLASSES):
            color = MASK_COLORS_RGBA[v]
            dummy = pg.PlotDataItem(
                [0], [0], pen=None, symbol="o",
                symbolBrush=pg.mkBrush(*color),
                symbolPen=pg.mkPen("k"),
                name=f"{v}: {name}",
            )
            self.plot_mask.addItem(dummy)
            dummy.setVisible(False)

        self.plot_model_mask = pg.PlotWidget(title="Fusion model mask (read-only suggestion)")
        self.plot_model_mask.setLabel("bottom", "Time", units="s")
        self.plot_model_mask.setLabel("left", "Class")
        self.plot_model_mask.showGrid(x=True, y=True, alpha=0.3)
        self.plot_model_mask.setYRange(-0.5, MASK_MAX + 0.5)
        ticks = [(i, f"{i}: {MASK_CLASSES[i]}") for i in range(len(MASK_CLASSES))]
        self.plot_model_mask.getAxis("left").setTicks([ticks])
        vbox.addWidget(self.plot_model_mask, stretch=2)

        self.model_curve = self.plot_model_mask.plot([], [], pen=pg.mkPen(width=2))
        self.model_stripe = pg.ImageItem()
        self.model_stripe.setLookupTable(self._class_lut)
        self.model_stripe.setOpts(axisOrder="row-major")
        self.model_stripe.setOpacity(0.35)
        self.plot_model_mask.addItem(self.model_stripe)

        self._model_hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self._model_hover_text = pg.TextItem("", anchor=(0, 1))
        self.plot_model_mask.addItem(self._model_hover_line)
        self.plot_model_mask.addItem(self._model_hover_text)
        self._model_hover_line.hide()
        self._model_hover_text.hide()

        self.plot_spec.setXLink(self.plot_signal)
        self.plot_mask.setXLink(self.plot_signal)
        self.plot_model_mask.setXLink(self.plot_signal)

        self.setCentralWidget(central)

    def _connect_signals(self):
        self.btn_load.clicked.connect(self.load_file)
        self.spin_fs.valueChanged.connect(self._on_params_changed)
        self.spin_win.valueChanged.connect(self._on_params_changed)
        self.btn_prev.clicked.connect(self.prev_window)
        self.btn_next.clicked.connect(self.next_window)
        self.spin_goto.valueChanged.connect(self.goto_window)
        self.btn_plus.clicked.connect(lambda: self._nudge_selected(+1))
        self.btn_minus.clicked.connect(lambda: self._nudge_selected(-1))
        self.home_btn.clicked.connect(self.reset_zoom)
        self.threshold_spin.valueChanged.connect(self.update_overlay_safe)

        self.select_btn.toggled.connect(self.toggle_select_mode)
        self.label_btn.toggled.connect(self.toggle_label_mode)

        self.btn_reload_model.clicked.connect(self._try_load_model)
        self.btn_browse_bundle.clicked.connect(self._browse_bundle)
        self.model_combo.currentTextChanged.connect(lambda _: self._try_load_model(auto=True))

        self.plot_model_mask.scene().sigMouseMoved.connect(self._on_model_mask_mouse_moved)

    # ---------------- Model selection ----------------
    def _list_bundle_dirs(self) -> List[str]:
        runs = CFG.runs_dir
        if not os.path.isdir(runs):
            return []

        items = []
        token = CFG.model_name_token.lower()
        for p in os.listdir(runs):
            full = os.path.join(runs, p)
            if os.path.isdir(full) and token in p.lower():
                items.append(p)
        return sorted(items)

    def _refresh_bundle_combo(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        items = self._list_bundle_dirs()
        self.model_combo.addItems(items)

        if CFG.default_bundle and CFG.default_bundle in items:
            self.model_combo.setCurrentText(CFG.default_bundle)
        elif items:
            self.model_combo.setCurrentIndex(0)

        self.model_combo.blockSignals(False)

    def _selected_bundle_path(self) -> Optional[str]:
        name = self.model_combo.currentText().strip() if hasattr(self, "model_combo") else ""
        if not name:
            return None
        path = os.path.join(CFG.runs_dir, name)
        return path if os.path.isdir(path) else None

    def _browse_bundle(self):
        start_dir = os.path.abspath(CFG.runs_dir) if os.path.isdir(CFG.runs_dir) else os.getcwd()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select fusion model bundle folder", start_dir)
        if not path:
            return

        name = os.path.basename(path)
        if CFG.model_name_token.lower() not in name.lower():
            QtWidgets.QMessageBox.warning(
                self,
                "Not a fusion bundle",
                f"Selected folder name does not contain '{CFG.model_name_token}'."
            )
            return

        runs_abs = os.path.abspath(CFG.runs_dir)
        path_abs = os.path.abspath(path)

        if os.path.commonpath([runs_abs, path_abs]) == runs_abs:
            rel = os.path.relpath(path_abs, runs_abs)
            if rel not in self._list_bundle_dirs():
                self.model_combo.addItem(rel)
            self.model_combo.setCurrentText(rel)
        else:
            self._try_load_model(bundle_override=path_abs)

    def _try_load_model(self, auto: bool = False, bundle_override: Optional[str] = None):
        bundle_dir = bundle_override or self._selected_bundle_path()
        if not bundle_dir:
            if not auto:
                self.lbl_model.setText("model: no fusion bundle selected")
            self.predictor = None
            self._model_ok = False
            return

        try:
            self.predictor = DLBundledPredictorFusion(bundle_dir, device_pref=CFG.device)
            self._model_ok = True
            b = self.predictor.bundle
            if b is not None:
                self.lbl_model.setText(
                    f"model: OK ({os.path.basename(bundle_dir)} | {b.arch} | fusion 1D+2D | dev={self.predictor.device.type})"
                )
            else:
                self.lbl_model.setText(f"model: OK ({os.path.basename(bundle_dir)})")
        except Exception as e:
            self.predictor = None
            self._model_ok = False
            self.lbl_model.setText(f"model: not loaded ({type(e).__name__}: {e})")
            print("Fusion bundle load error:", repr(e))

        if self.eeg is not None:
            self._render_current_window()

    # ---------------- Windowing ----------------
    def _num_windows(self) -> int:
        assert self.eeg is not None
        n_win = int(np.floor((len(self.eeg) / self.fs) / self.window_size_s))
        return max(n_win, 0)

    def _window_bounds_samples(self) -> Tuple[int, int]:
        start_s = self.window_idx * self.window_size_s
        end_s = start_s + self.window_size_s
        a = int(round(start_s * self.fs))
        b = int(round(end_s * self.fs))
        return a, b

    def _window_time_bounds_seconds(self) -> Tuple[float, float]:
        start_s = self.window_idx * self.window_size_s
        end_s = start_s + self.window_size_s
        return float(start_s), float(end_s)

    def _json_path(self) -> str:
        assert self.path_npy is not None
        base = os.path.splitext(self.path_npy)[0]
        return base + "_mask.json"

    def _window_key(self) -> str:
        start_s, end_s = self._window_time_bounds_seconds()
        return f"{start_s:.3f}-{end_s:.3f}"

    def _load_mask_from_json_if_exists(self) -> Optional[np.ndarray]:
        jp = self._json_path()
        if not os.path.exists(jp):
            return None

        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        key = self._window_key()
        win = data.get("windows", {}).get(key, None)
        if not win:
            return None

        mask_list = win.get("mask", None)
        t_spec_list = win.get("t_spec", None)
        if mask_list is None or t_spec_list is None:
            return None

        mask_arr = np.asarray(mask_list, dtype=float)
        t_arr = np.asarray(t_spec_list, dtype=float)
        if len(t_arr) != len(self.t_spec):
            return None

        return mask_arr

    def _write_json(self):
        assert self.path_npy is not None
        jp = self._json_path()
        payload: Dict[str, Any] = {}

        if os.path.exists(jp):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        if "recording" not in payload:
            payload["recording"] = os.path.basename(self.path_npy)

        payload["fs_hz"] = float(self.fs)
        payload["windows"] = payload.get("windows", {})

        key = self._window_key()
        start_s, end_s = self._window_time_bounds_seconds()

        payload["windows"][key] = {
            "window_index": int(self.window_idx),
            "window_start_s": float(start_s),
            "window_end_s": float(end_s),
            "window_size_s": float(self.window_size_s),
            "fs_hz": float(self.fs),
            "t_spec": self.t_spec.tolist(),
            "mask": self.mask.tolist(),
        }

        tmp = jp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, jp)

    # ---------------- Controls ----------------
    def load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select 1D EEG .npy file", "", "NumPy files (*.npy)")
        if not path:
            return

        arr = np.load(path)
        if arr.ndim != 1:
            QtWidgets.QMessageBox.critical(self, "Error", "The .npy file must contain a 1D array.")
            return

        self.path_npy = path
        self.eeg = np.asarray(arr, dtype=float)

        self.fs = float(self.spin_fs.value())
        self.window_size_s = float(self.spin_win.value())
        self.window_idx = 0

        self._dirty = False
        self._original_mask = None

        self._render_current_window()

    def _on_params_changed(self):
        if self.eeg is None:
            return
        self.fs = float(self.spin_fs.value())
        self.window_size_s = float(self.spin_win.value())
        self.window_idx = int(np.clip(self.window_idx, 0, max(0, self._num_windows() - 1)))
        self._render_current_window()

    def prev_window(self):
        if self.eeg is None:
            return
        if self.window_idx > 0:
            self.window_idx -= 1
            self._render_current_window()

    def next_window(self):
        if self.eeg is None:
            return
        if self.window_idx < self._num_windows() - 1:
            self.window_idx += 1
            self._render_current_window()

    def goto_window(self, one_based_index: int):
        if self.eeg is None:
            return
        nwin = self._num_windows()
        if nwin <= 0:
            return
        idx0 = int(np.clip(one_based_index - 1, 0, nwin - 1))
        if idx0 == self.window_idx:
            return
        self.window_idx = idx0
        self._render_current_window()

    def reset_zoom(self):
        if self.t_sig is not None and len(self.t_sig) > 1:
            self.plot_signal.enableAutoRange()
            self.plot_spec.enableAutoRange()
            self.plot_mask.enableAutoRange()
            self.plot_model_mask.enableAutoRange()

    def toggle_select_mode(self, enabled: bool):
        self._rb_zoom_active = bool(enabled)
        if enabled:
            self.label_btn.setChecked(False)
            self.plot_signal.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.plot_signal.viewport().unsetCursor()

    def toggle_label_mode(self, enabled: bool):
        self._rb_label_active = bool(enabled)
        if enabled:
            self.select_btn.setChecked(False)
            self.plot_signal.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.plot_signal.viewport().unsetCursor()

    # ---------------- Rendering ----------------
    def _render_current_window(self):
        assert self.eeg is not None
        nwin = self._num_windows()
        if nwin <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Signal too short for the current window size / fs.")
            return

        self.window_idx = int(np.clip(self.window_idx, 0, nwin - 1))

        a, b = self._window_bounds_samples()
        b = min(b, len(self.eeg))

        self.sig_win = self.eeg[a:b].copy()
        self.sig_win = self.sig_win - np.median(self.sig_win)

        sqrt_med = np.sqrt(np.median(self.sig_win ** 2))
        sqrt_med = max(float(sqrt_med), 1e-8)
        factor = 25.0 / sqrt_med
        if factor <= 1.0:
            self.sig_win = self.sig_win * factor

        self.t_sig = np.arange(len(self.sig_win), dtype=float) / float(self.fs)

        f, t, Sxx = spectrogram(
            self.sig_win,
            self.fs,
            nperseg_factor=CFG.nperseg_factor,
            noverlap_factor=CFG.noverlap_factor,
            nfft_factor=CFG.nfft_factor,
            f_cut=CFG.f_cut_hz,
        )

        self.f = np.asarray(f, dtype=float)
        self.t_spec = np.asarray(t, dtype=float)
        self.Sxx = np.asarray(Sxx, dtype=float)

        if self.Sxx.ndim != 2:
            QtWidgets.QMessageBox.critical(self, "Error", f"Sxx must be 2D, got ndim={self.Sxx.ndim}")
            return
        if self.Sxx.shape == (len(self.t_spec), len(self.f)):
            self.Sxx = self.Sxx.T
        if self.Sxx.shape != (len(self.f), len(self.t_spec)):
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Sxx shape mismatch: got {self.Sxx.shape}, expected ({len(self.f)}, {len(self.t_spec)})"
            )
            return

        self.model_mask = np.zeros(len(self.t_spec), dtype=int)
        if self.predictor is not None and len(self.t_spec) > 0 and self.Sxx is not None:
            t0 = time()
            try:
                y_pred = self.predictor.predict_mask_from_signal_and_sxx(
                    self.sig_win,
                    self.Sxx,
                    out_T=len(self.t_spec),
                )
                self.model_mask = np.asarray(np.clip(y_pred, 0, MASK_MAX), dtype=int)
            except Exception as e:
                self.lbl_model.setText(f"model: inference failed ({e})")
                self.model_mask = np.zeros(len(self.t_spec), dtype=int)
            t1 = time()
            print(f'time for prediction {t1 - t0}')
        self.mask = np.zeros(len(self.t_spec), dtype=int)
        loaded = self._load_mask_from_json_if_exists()
        if loaded is not None and len(loaded) == len(self.mask):
            self.mask = np.asarray(np.clip(np.round(loaded), 0, MASK_MAX), dtype=int)

        self._original_mask = self.mask.copy()
        self._dirty = False

        self._update_window_nav_ui()
        self._update_plots()
        self.update_overlay_safe()
        self._update_status_line()

    def _update_plots(self):
        self.curve_sig.setData(self.t_sig, self.sig_win)
        self._update_spectrogram_image()
        self._update_mask_plot()
        self._update_model_mask_plot()

        self.plot_signal.setXRange(0.0, float(self.window_size_s), padding=0.02)
        self.plot_mask.setYRange(-0.5, MASK_MAX + 0.5, padding=0.02)
        self.plot_model_mask.setYRange(-0.5, 11, padding=0.02)

    def _update_spectrogram_image(self):
        if self.Sxx is None:
            return

        Sxx_db = 10.0 * np.log10(np.maximum(self.Sxx, 1e-20))
        finite = np.isfinite(Sxx_db)
        if np.any(finite):
            pdown, pup = np.percentile(Sxx_db[finite], [50, 100])
            self.img_item.setImage(Sxx_db, autoLevels=False, levels=(float(pdown) - 1, float(pup) + 1))
        else:
            self.img_item.setImage(Sxx_db, autoLevels=True)

        dt = float(self.t_spec[1] - self.t_spec[0]) if len(self.t_spec) > 1 else 1.0 / float(self.fs)
        df = float(self.f[1] - self.f[0]) if len(self.f) > 1 else float(self.fs) / 2.0
        x0 = float(self.t_spec[0]) if len(self.t_spec) else 0.0
        y0 = float(self.f[0]) if len(self.f) else 0.0

        tr = QtGui.QTransform()
        tr.translate(x0, y0)
        tr.scale(dt, df)
        self.img_item.setTransform(tr)

        width = dt * Sxx_db.shape[1]
        height = df * Sxx_db.shape[0]
        self.plot_spec.setXRange(x0, x0 + width, padding=0)
        self.plot_spec.setYRange(y0, y0 + height, padding=0)

    def _update_mask_plot(self):
        for p in self.mask_points:
            try:
                self.plot_mask.removeItem(p)
            except Exception:
                pass
        self.mask_points = []

        if self.t_spec is None or len(self.t_spec) == 0:
            self.mask_curve.setData([], [])
            return

        self.mask_curve.setData(self.t_spec, self.mask.astype(float))
        for i, (tx, yy) in enumerate(zip(self.t_spec, self.mask)):
            pt = DraggableMaskPoint(x=float(tx), y=float(yy), index=i, viewer=self)
            self.plot_mask.addItem(pt)
            self.mask_points.append(pt)

    def _update_model_mask_plot(self):
        if self.t_spec is None or len(self.t_spec) == 0 or self.model_mask is None or len(self.model_mask) == 0:
            self.model_curve.setData([], [])
            self.model_stripe.setImage(np.zeros((1, 1)), autoLevels=True)
            return

        self.model_curve.setData(self.t_spec, self.model_mask.astype(float))

        stripe = self.model_mask[np.newaxis, :].astype(float)
        stripe_scaled = stripe * (255.0 / max(1, MASK_MAX))
        self.model_stripe.setImage(stripe_scaled, autoLevels=False, levels=(0, 255))

        dt = float(self.t_spec[1] - self.t_spec[0]) if len(self.t_spec) > 1 else 1.0 / float(self.fs)
        x0 = float(self.t_spec[0])
        y0 = -0.5
        tr = QtGui.QTransform()
        tr.translate(x0, y0)
        tr.scale(dt, (MASK_MAX + 1.0))
        self.model_stripe.setTransform(tr)

    # ---------------- Hover ----------------
    def _on_model_mask_mouse_moved(self, pos):
        if self.t_spec is None or len(self.t_spec) == 0 or self.model_mask is None or len(self.model_mask) == 0:
            self._model_hover_line.hide()
            self._model_hover_text.hide()
            return

        vb = self.plot_model_mask.getPlotItem().vb
        mousePoint = vb.mapSceneToView(pos)
        x = float(mousePoint.x())

        xr = self.plot_model_mask.viewRange()[0]
        if x < xr[0] or x > xr[1]:
            self._model_hover_line.hide()
            self._model_hover_text.hide()
            return

        idx = int(np.clip(np.searchsorted(self.t_spec, x), 0, len(self.t_spec) - 1))
        if idx > 0 and abs(self.t_spec[idx] - x) > abs(self.t_spec[idx - 1] - x):
            idx -= 1

        cls = int(np.clip(int(self.model_mask[idx]), 0, len(MASK_CLASSES) - 1))
        name = MASK_CLASSES[cls]

        self._model_hover_line.setPos(self.t_spec[idx])
        self._model_hover_text.setText(f"{cls}: {name}")
        self._model_hover_text.setPos(self.t_spec[idx], MASK_MAX + 0.4)

        self._model_hover_line.show()
        self._model_hover_text.show()

    # ---------------- Overlay ----------------
    def update_overlay_safe(self):
        if self.Sxx is None or self.f is None or self.t_spec is None:
            return

        thr = float(self.threshold_spin.value())
        self.threshold_line_item.setPos(thr)

        try:
            from Functions.edge_frequency import edge_frequencies_significant_value
            ef = edge_frequencies_significant_value(self.Sxx, self.f, max_val=50, threshold=thr)[0]
            self.overlay_curve_item.setData(self.t_spec, ef)
            self.overlay_curve_item.show()
        except Exception:
            self.overlay_curve_item.hide()

    # ---------------- EEG -> mask labeling ----------------
    def _apply_eeg_label_to_mask(self, start_s: float, end_s: float):
        if self.t_spec is None or len(self.t_spec) == 0:
            return

        start_s = float(np.clip(start_s, 0.0, self.window_size_s))
        end_s = float(np.clip(end_s, 0.0, self.window_size_s))
        if end_s <= start_s:
            return

        text = self.class_combo.currentText()
        try:
            class_val = int(text.split(":")[0].strip())
        except Exception:
            class_val = 0
        class_val = int(np.clip(class_val, 0, MASK_MAX))

        i0 = int(np.argmin(np.abs(self.t_spec - start_s)))
        i1 = int(np.argmin(np.abs(self.t_spec - end_s)))
        lo, hi = min(i0, i1), max(i0, i1)
        idxs = [lo] if lo == hi else list(range(lo, hi + 1))

        self.mask[idxs] = class_val

        self.mask_curve.setData(self.t_spec, self.mask.astype(float))
        for idx in idxs:
            p = self.mask_points[idx]
            p.setPos(QPointF(p.fixed_x, float(class_val)))
            p.set_selected(p.selected)

        self._mark_dirty_and_save_if_needed()

    # ---------------- Mask editing ----------------
    def _mask_changed(self) -> bool:
        if self._original_mask is None:
            return False
        if len(self.mask) != len(self._original_mask):
            return True
        return not np.array_equal(self.mask, self._original_mask)

    def _set_mask_value(self, index: int, new_y: int):
        if index is None or index < 0 or index >= len(self.mask):
            return
        new_y = int(np.clip(new_y, 0, MASK_MAX))
        if int(self.mask[index]) == new_y:
            return

        self.mask[index] = new_y
        self.mask_curve.setData(self.t_spec, self.mask.astype(float))

        p = self.mask_points[index]
        p.set_selected(p.selected)
        self._mark_dirty_and_save_if_needed()

    def _group_set_selected_mask(self, new_y: int):
        if self._updating_group:
            return

        self._updating_group = True
        try:
            changed = False
            new_y = int(np.clip(new_y, 0, MASK_MAX))

            for p in self.mask_points:
                if p.selected:
                    idx = p.index
                    if int(self.mask[idx]) != new_y:
                        self.mask[idx] = new_y
                        p.setPos(QPointF(p.fixed_x, float(new_y)))
                        p.set_selected(True)
                        changed = True

            if changed:
                self.mask_curve.setData(self.t_spec, self.mask.astype(float))
                self._mark_dirty_and_save_if_needed()
        finally:
            self._updating_group = False

    def _nudge_selected(self, delta: int):
        if self.eeg is None or len(self.mask) == 0:
            return

        selected = [p for p in self.mask_points if p.selected]
        if not selected:
            return

        changed = False
        for p in selected:
            idx = p.index
            new_y = int(np.clip(int(self.mask[idx]) + int(delta), 0, MASK_MAX))
            if int(self.mask[idx]) != new_y:
                self.mask[idx] = new_y
                p.setPos(QPointF(p.fixed_x, float(new_y)))
                p.set_selected(True)
                changed = True

        if changed:
            self.mask_curve.setData(self.t_spec, self.mask.astype(float))
            self._mark_dirty_and_save_if_needed()

    def _mark_dirty_and_save_if_needed(self):
        if self._mask_changed():
            self._dirty = True
            self._write_json()
        else:
            self._dirty = False
        self._update_status_line()

    # ---------------- Status ----------------
    def _update_status_line(self):
        if self.eeg is None or self.path_npy is None:
            self.lbl_status.setText("No file loaded.")
            self.lbl_dirty.setText("")
            return

        self._update_window_nav_ui()
        start_s, end_s = self._window_time_bounds_seconds()
        nwin = self._num_windows()
        sel = sum(1 for p in self.mask_points if p.selected)

        self.lbl_status.setText(
            f"{os.path.basename(self.path_npy)} | window {self.window_idx + 1}/{nwin} "
            f"| [{start_s:.2f}, {end_s:.2f}] s | selected mask pts={sel}"
        )
        self.lbl_dirty.setText("● saved (modified)" if self._dirty else "")

    def _update_window_nav_ui(self):
        if self.eeg is None:
            self.lbl_winpos.setText("- / -")
            self.spin_goto.blockSignals(True)
            self.spin_goto.setRange(1, 1)
            self.spin_goto.setValue(1)
            self.spin_goto.blockSignals(False)
            return

        nwin = self._num_windows()
        if nwin <= 0:
            self.lbl_winpos.setText("0 / 0")
            self.spin_goto.blockSignals(True)
            self.spin_goto.setRange(1, 1)
            self.spin_goto.setValue(1)
            self.spin_goto.blockSignals(False)
            return

        self.lbl_winpos.setText(f"{self.window_idx + 1} / {nwin}")
        self.spin_goto.blockSignals(True)
        self.spin_goto.setRange(1, nwin)
        self.spin_goto.setValue(self.window_idx + 1)
        self.spin_goto.blockSignals(False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = EEGLabelerWithModelFusion()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
