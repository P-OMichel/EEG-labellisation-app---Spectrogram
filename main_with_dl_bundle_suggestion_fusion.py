# -*- coding: utf-8 -*-
"""
Merged App: EEG labeling (editable mask + autosave) + Model suggestion mask (read-only)

You provided:
- main.py: labeling app with editable mask points + JSON saving fileciteturn6file0
- eeg_mask_inference_viewer.py: inference app that loads XGBoost + extracts features fileciteturn6file1

What this merged app does:
- Keeps your labeling workflow:
    * EEG plot (top)
    * Spectrogram (middle)
    * Editable mask plot (bottom #1) with draggable points and EEG-rectangle->mask labeling
    * Autosave per-window to JSON exactly like your main.py
- Adds a second mask plot BELOW (bottom #2):
    * Model-predicted mask (read-only) for the same window
    * Same time axis (t_spec)
    * Colored stripe + line
    * Cannot be edited
- When you label/correct the editable mask, the model mask stays unchanged (visual reference only)

IMPORTANT:
- You said you'll "put and load the model in the app code directly":
  Edit CFG.model_path / CFG.scaler_path below.
- This app expects your feature extraction functions available as:
    from feature_extractor import extract_features_time_series, extract_features_spectrogram
  (same as in your inference script).
"""

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


# ---------------------------
# CONFIG | choose default parameters 
# ---------------------------
@dataclass
class AppConfig:
    # ---- UI defaults ----
    fs_default: float = 128.0
    window_size_s_default: float = 30.0

    # ---- Model bundles ----
    # A bundle is a folder containing: config.json, model.pt, stats.json
    runs_dir: str = "DL/runs"
    default_bundle: Optional[str] = None  # if None, picks the first available in runs_dir
    device: str = "cud"  # will fall back to cpu if unavailable

    # Optional lags | NOTE: should be same as for training
    lags: tuple[int, ...] = (1, 2)

    # Spectrogram settings | NOTE: should be same as for training 
    f_cut_hz: float = 45.0
    nperseg_factor: float = 1.0
    noverlap_factor: float = 0.90
    nfft_factor: float = 1.0



CFG = AppConfig()



# ---------------------------
# Spectrogram (window-local)
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
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
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

    # cut at f_cut
    if len(f_spectro) > 1:
        df = f_spectro[1] - f_spectro[0]
        j = int(f_cut / df)
        j = max(1, min(j, len(f_spectro)))
        f_spectro = f_spectro[:j]
        Sxx = Sxx[:j, :]

    return f_spectro, t_spectro, Sxx


def make_jet_lut(n: int = 256) -> np.ndarray:
    try:
        from matplotlib import cm  # type: ignore
        cmap = cm.get_cmap("jet", n)  # warning ok
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
MASK_CLASSES = ["ok", "alpha-sup", "IES", "gc", "shallow", "gamma", "eye artifact", "HF artifact", "large artifact", "awake"]
MASK_COLORS_RGBA = {
    0: (80, 200, 120, 220),   # ok
    1: (120, 120, 255, 220),  # alpha-sup
    2: (255, 170, 0, 220),    # IES
    3: (30, 30, 30, 220),     # gc
    4: (200, 200, 0, 220),    # shallow
    5: (255, 0, 200, 220),    # gamma
    6: (0, 220, 220, 220),    # eye artifact
    7: (255, 0, 0, 220),      # HF artifact
    8: (150, 75, 0, 220),     # large artifact
    9: (100, 75, 0, 220),     # awake
}
MASK_MAX = max(MASK_COLORS_RGBA.keys())


import torch
import torch.nn.functional as F

def _resize_2d_linear(x: np.ndarray, out_F: int, out_T: int) -> np.ndarray:
    """
    Resize a 2D numpy array (F, T) to (out_F, out_T)
    using bilinear interpolation (PyTorch backend).
    """
    if x.ndim != 2:
        raise ValueError("Expected 2D array (F,T)")

    xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
    xt = F.interpolate(
        xt,
        size=(out_F, out_T),
        mode="bilinear",
        align_corners=False
    )
    return xt.squeeze(0).squeeze(0).cpu().numpy()

# ---------------------------
# Mask plot helpers (editable mask plot)
# ---------------------------
class RubberbandMaskPlot(pg.PlotWidget):
    """Shift + drag rectangle to select points. Ctrl+Shift drag to unselect."""
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
# EEG Rubberband selection / label on top plot
# ---------------------------
class EEGLabelRubberBandFilter(QtCore.QObject):
    """
    Event filter installed on plot_signal.viewport().
    Two modes controlled by viewer:
      - viewer._rb_zoom_active
      - viewer._rb_label_active
    Draws rectangle and on release either zoom or apply label.
    """
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

            # ignore tiny drags
            if rect.width() < 4 or rect.height() < 4:
                return True

            # map to data coords
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
# DL bundle predictor (HF-like)
# ---------------------------
# This app should NOT define architectures. It loads a trained model bundle from `runs/<bundle>/`
# and reconstructs the architecture via your `src/` package (registry), then loads weights.

from types import SimpleNamespace
import importlib
import pkgutil

def _import_all_src_models():
    """Import all modules under src.models so @register_model decorators populate the registry."""
    try:
        import DL.src.models  # noqa: F401
        pkg = DL.src.models
        for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(m.name)
    except Exception:
        # If src isn't importable, user is not running from repo root.
        pass

_import_all_src_models()

class ModelBundle(NamedTuple):
    model: "torch.nn.Module"
    mean: float
    std: float
    spec_F: int
    spec_T: int
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


class DLBundledPredictor:
    """Loads a DL model bundle from runs/ using src.io.bundle (preferred)."""

    def __init__(self, bundle_dir: str, device_pref: str = "cuda"):
        import torch
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
        import torch

        # Prefer your project loader
        try:
            from DL.src.io.bundle import load_bundle  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import src.io.bundle.load_bundle. "
                "Run this app from the repo root (same level as src/)."
            ) from e

        # Call load_bundle with a few common signatures
        try:
            loaded = load_bundle(self.bundle_dir, device=self._device)
        except TypeError:
            try:
                loaded = load_bundle(self.bundle_dir, device_str=self._device.type)
            except TypeError:
                try:
                    loaded = load_bundle(self.bundle_dir, map_location=self._device)
                except TypeError:
                    loaded = load_bundle(self.bundle_dir)

        # The loader may return a dict or a small object. Normalize.
        ld = _as_dict(loaded)
        print(ld)

        model = _get_first(loaded, ["model"], None) or ld.get("model")
        if model is None:
            raise ValueError("load_bundle(...) did not return a model")

        model.to(self._device)
        model.eval()

        stats = _get_first(loaded, ["stats"], None)
        if stats is None:
            raise ValueError("load_bundle(...) did not return stats for normalization")        
        mean = float(_get_first(stats, ["mean"], 0))
        std = float(_get_first(stats, ["std"], 1))
        spec_F = int(_get_first(loaded, ["spec_F", "F"], ld.get("spec_F", 45)))
        spec_T = int(_get_first(loaded, ["spec_T", "T"], ld.get("spec_T", 297)))
        num_classes = int(_get_first(loaded, ["num_classes", "C"], ld.get("num_classes", 10)))
        arch = str(_get_first(loaded, ["arch", "model_name", "name"], ld.get("arch", "unknown")))

        self.bundle = ModelBundle(
            model=model,
            mean=mean,
            std=std,
            spec_F=spec_F,
            spec_T=spec_T,
            num_classes=num_classes,
            arch=arch,
            bundle_dir=self.bundle_dir,
        )

    def predict_mask_from_sxx(self, Sxx: np.ndarray, out_T: int) -> np.ndarray:
        """
        Sxx: (F,T) power spectrogram for current window (f<=f_cut)
        out_T: number of time bins to return (len(t_spec))
        Returns: (out_T,) int mask
        """
        import torch
        if self.bundle is None:
            raise RuntimeError("Bundle not loaded")

        # Match training preprocessing: use log10(power) then global z-norm (mean/std from stats.json)
        x = np.log1p(Sxx + 0.00000000001)
        x = (x - self.bundle.mean) / self.bundle.std

        # Resize to model expected shape
        print(np.shape(x))
        x = _resize_2d_linear(x, self.bundle.spec_F, self.bundle.spec_T)  # (F,T)
        print(np.shape(x))
        xt = torch.from_numpy(x[None, None, :, :]).to(self._device)       # (1,1,F,T)

        with torch.no_grad():
            y = self.bundle.model(xt)
            print('np.shape(y)', np.shape(y))

        # Normalize output to logits (1,C,T_model)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.ndim == 3:
            # could be (1,T,C) or (1,C,T)
            if y.shape[1] == self.bundle.num_classes:
                logits = y
            elif y.shape[2] == self.bundle.num_classes:
                logits = y.permute(0, 2, 1)
            else:
                logits = y.permute(0, 2, 1)
        elif y.ndim == 4:
            logits = y.mean(dim=2)
        else:
            raise ValueError(f"Unexpected model output shape: {tuple(y.shape)}")

        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(int)

        # Resample to out_T if needed (nearest)
        if pred.shape[0] != out_T:
            x_old = np.linspace(0.0, 1.0, pred.shape[0])
            x_new = np.linspace(0.0, 1.0, out_T)
            idx = np.clip(np.round(np.interp(x_new, x_old, np.arange(pred.shape[0]))).astype(int), 0, pred.shape[0]-1)
            pred = pred[idx]

        return pred


# ---------------------------
# Main app

# ---------------------------
class EEGLabelerWithModel(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG labeling + model suggestion (read-only)")
        self.resize(950, 1050)

        # Fixed fs UI default
        self.fs: float = CFG.fs_default

        # Data state
        self.eeg: Optional[np.ndarray] = None
        self.path_npy: Optional[str] = None

        self.window_size_s: float = CFG.window_size_s_default
        self.window_idx: int = 0

        # current window arrays (time starts at 0 each window)
        self.t_sig: np.ndarray = np.array([])
        self.sig_win: np.ndarray = np.array([])

        # spectrogram (window time bins)
        self.t_spec: np.ndarray = np.array([])
        self.f: np.ndarray = np.array([])
        self.Sxx: Optional[np.ndarray] = None

        # editable mask aligned to t_spec
        self.mask: np.ndarray = np.array([])
        self._original_mask: Optional[np.ndarray] = None
        self._dirty: bool = False

        # model mask aligned to t_spec (read-only)
        self.model_mask: np.ndarray = np.array([])
        self._model_ok: bool = False
        self.predictor: Optional[DLBundledPredictor] = None

        # mask items
        self.mask_points: List[DraggableMaskPoint] = []
        self._updating_group = False

        # overlay
        self.overlay_curve_item = None
        self.threshold_line_item = None

        # modes for EEG rubberband
        self._rb_zoom_active = False
        self._rb_label_active = False

        # LUT for spectrogram + model mask stripe
        self._jet_lut = make_jet_lut(256)
        self._class_lut = make_class_lut(MASK_COLORS_RGBA, MASK_MAX, 256)

        self._build_ui()
        self._connect_signals()

        # install EEG rubberband filter
        self._eeg_rb_filter = EEGLabelRubberBandFilter(self)
        self.plot_signal.viewport().installEventFilter(self._eeg_rb_filter)

        # shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Left"), self, activated=self.prev_window)
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self, activated=self.next_window)
        QtGui.QShortcut(QtGui.QKeySequence("Up"), self, activated=lambda: self._nudge_selected(+1))
        QtGui.QShortcut(QtGui.QKeySequence("Down"), self, activated=lambda: self._nudge_selected(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+G"), self, activated=lambda: self.spin_goto.setFocus())

        # populate bundle dropdown from runs/ then load
        self._refresh_bundle_combo()
        self._try_load_model(auto=True)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # Controls row
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

        # zoom + label buttons
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

        # threshold for overlay
        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 50.0)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setSingleStep(1.0)
        self.threshold_spin.setValue(10.0)
        self.threshold_spin.setFixedWidth(90)
        controls.addWidget(self.threshold_spin)

        # file loaded
        controls.addStretch(1)
        self.lbl_status = QtWidgets.QLabel("No file loaded.")
        self.lbl_status.setMinimumWidth(280)
        controls.addWidget(self.lbl_status)

        vbox.addLayout(controls)
        

        # Mask nudge row
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

        # model bundle selection + reload
        controls_2.addSpacing(16)
        controls_2.addWidget(QtWidgets.QLabel("Model bundle:"))
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

        # Plots
        self.plot_signal = pg.PlotWidget(title="EEG (current window)")
        self.plot_signal.setLabel("bottom", "Time", units="s")
        self.plot_signal.setLabel("left", "Amplitude")
        self.plot_signal.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_signal, stretch=2)
        self.curve_sig = self.plot_signal.plot([], [], pen=pg.mkPen(width=1))

        self.plot_spec = pg.PlotWidget(title="Spectrogram")
        self.plot_spec.setLabel("bottom", "Time", units="s")
        self.plot_spec.setLabel("left", "Frequency", units="Hz")
        self.plot_spec.showGrid(x=True, y=True, alpha=0.3)
        vbox.addWidget(self.plot_spec, stretch=3)

        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(self._jet_lut)
        self.img_item.setOpts(axisOrder="row-major")
        self.plot_spec.addItem(self.img_item)

        # overlay curve + threshold line
        self.overlay_curve_item = pg.PlotDataItem(pen=pg.mkPen(width=2))
        self.plot_spec.addItem(self.overlay_curve_item)
        self.threshold_line_item = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_spec.addItem(self.threshold_line_item)

        # --- Editable mask plot (as in main.py) ---
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

        # --- Model mask plot (read-only) ---
        self.plot_model_mask = pg.PlotWidget(title="Model mask (read-only suggestion)")
        self.plot_model_mask.setLabel("bottom", "Time", units="s")
        self.plot_model_mask.setLabel("left", "Class")
        self.plot_model_mask.showGrid(x=True, y=True, alpha=0.3)
        self.plot_model_mask.setYRange(-0.5, MASK_MAX + 0.5)
        # Y ticks with class names
        ticks = [(i, f"{i}: {MASK_CLASSES[i]}") for i in range(len(MASK_CLASSES))]
        self.plot_model_mask.getAxis("left").setTicks([ticks])
        vbox.addWidget(self.plot_model_mask, stretch=2)

        self.model_curve = self.plot_model_mask.plot([], [], pen=pg.mkPen(width=2))
        self.model_stripe = pg.ImageItem()
        self.model_stripe.setLookupTable(self._class_lut)
        self.model_stripe.setOpts(axisOrder="row-major")
        self.model_stripe.setOpacity(0.35)
        self.plot_model_mask.addItem(self.model_stripe)

        # Hover on model mask: vertical cursor + class text
        self._model_hover_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self._model_hover_text = pg.TextItem("", anchor=(0, 1))
        self.plot_model_mask.addItem(self._model_hover_line)
        self.plot_model_mask.addItem(self._model_hover_text)
        self._model_hover_line.hide()
        self._model_hover_text.hide()

        # link x
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

        # Hover on model mask plot
        self.plot_model_mask.scene().sigMouseMoved.connect(self._on_model_mask_mouse_moved)

    # ---------------- Model ----------------
    
    # ---------------- Model ----------------
    def _list_bundle_dirs(self) -> List[str]:
        runs = CFG.runs_dir
        if not os.path.isdir(runs):
            return []
        return sorted([p for p in os.listdir(runs) if os.path.isdir(os.path.join(runs, p))])

    def _refresh_bundle_combo(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        items = self._list_bundle_dirs()
        self.model_combo.addItems(items)
        # default selection
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
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select model bundle folder", start_dir)
        if not path:
            return
        # If user selected a bundle inside runs/, add/select it. Otherwise load directly.
        runs_abs = os.path.abspath(CFG.runs_dir)
        path_abs = os.path.abspath(path)
        if os.path.commonpath([runs_abs, path_abs]) == runs_abs:
            rel = os.path.relpath(path_abs, runs_abs)
            if rel not in self._list_bundle_dirs():
                # still add it if it's directly under runs
                self.model_combo.addItem(rel)
            self.model_combo.setCurrentText(rel)
        else:
            # load directly (not in runs/)
            self._try_load_model(bundle_override=path_abs)

    def _try_load_model(self, auto: bool = False, bundle_override: Optional[str] = None):
        bundle_dir = bundle_override or self._selected_bundle_path()
        if not bundle_dir:
            if not auto:
                self.lbl_model.setText("model: no bundle selected")
            self.predictor = None
            self._model_ok = False
            return
        try:
            self.predictor = DLBundledPredictor(bundle_dir, device_pref=CFG.device)
            self._model_ok = True
            b = self.predictor.bundle
            if b is not None:
                self.lbl_model.setText(
                    f"model: OK ({os.path.basename(bundle_dir)} | {b.arch} | F×T={b.spec_F}×{b.spec_T} | dev={self.predictor.device.type})"
                )
            else:
                self.lbl_model.setText(f"model: OK ({os.path.basename(bundle_dir)})")
        except Exception as e:
            self.predictor = None
            self._model_ok = False
            if not auto:
                self.lbl_model.setText(f"model: not loaded ({e})")

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
        self.t_sig = np.arange(len(self.sig_win), dtype=float) / float(self.fs)

        # ---  ESTIMATE IF REDUCTION SCALING OF EEG IS NECESSARY ---
        sqrt_med = np.sqrt(np.median(self.sig_win**2))
        factor = 25 / sqrt_med
        # NOTE: uncomment to have only segments of high amplitude normalised
        if factor <= 1:
            self.sig_win = self.sig_win * factor

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

        # ensure (len(f), len(t))
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

        # --- model mask (read-only) ---
        self.model_mask = np.zeros(len(self.t_spec), dtype=int)
        if self.predictor is not None and len(self.t_spec) > 0 and self.Sxx is not None:
            try:
                y_pred = self.predictor.predict_mask_from_sxx(self.Sxx, out_T=len(self.t_spec))
                self.model_mask = np.asarray(np.clip(y_pred, 0, MASK_MAX), dtype=int)
            except Exception as e:
                self.lbl_model.setText(f"model: inference failed ({e})")
                self.model_mask = np.zeros(len(self.t_spec), dtype=int)

        # --- editable mask load/init ---
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
        # signal
        self.curve_sig.setData(self.t_sig, self.sig_win)

        # spectrogram
        self._update_spectrogram_image()

        # editable mask
        self._update_mask_plot()

        # model mask
        self._update_model_mask_plot()

        # x/y range
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
        # remove old points
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


    # ---------------- Hover (model mask) ----------------
    def _on_model_mask_mouse_moved(self, pos):
        """Show class name at mouse x-position on the model mask plot."""
        if self.t_spec is None or len(self.t_spec) == 0 or self.model_mask is None or len(self.model_mask) == 0:
            if hasattr(self, "_model_hover_line"):
                self._model_hover_line.hide()
                self._model_hover_text.hide()
            return

        vb = self.plot_model_mask.getPlotItem().vb
        mousePoint = vb.mapSceneToView(pos)
        x = float(mousePoint.x())

        # only show if inside current x-range
        xr = self.plot_model_mask.viewRange()[0]
        if x < xr[0] or x > xr[1]:
            self._model_hover_line.hide()
            self._model_hover_text.hide()
            return

        # nearest spectrogram bin
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

    # ---------------- Overlay (safe) ----------------
    def update_overlay_safe(self):
        """If your external Functions.edge_frequency exists, plot ef curve. Otherwise hide overlay."""
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

    # ---------------- Label EEG -> mask ----------------
    def _apply_eeg_label_to_mask(self, start_s: float, end_s: float):
        """Convert EEG-time interval to nearest t_spec indices and set mask to selected class value."""
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
        lo, hi = (min(i0, i1), max(i0, i1))
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

    # ---------------- Status / nav ----------------
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
    w = EEGLabelerWithModel()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
