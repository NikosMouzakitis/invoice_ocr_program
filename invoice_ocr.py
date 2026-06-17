import pygame
import sys
import os
import json
import subprocess
import tempfile
import numpy as np
from PIL import Image
import pyperclip
from tkinter import filedialog, Tk
import tkinter as tk
from tkinter import ttk
import cv2
import pytesseract
import threading

# pip install pygame pillow pyperclip numpy opencv-python-headless pytesseract

SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 800
TOOLBAR_H     = 52
IMG_AREA_H    = SCREEN_HEIGHT - TOOLBAR_H

WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
LIGHT_GRAY = (210, 210, 210)
YELLOW     = (255, 210,   0)
CYAN       = (0,   200, 210)
GREEN      = (60,  200,  80)
TOOLBAR_BG = (22,  27,  36)
BTN_NORMAL = (48,  54,  68)
BTN_HOVER  = (68,  78,  98)
BTN_TEXT   = (228, 233, 240)
BTN_GREEN  = (30,  110,  50)
BTN_GREEN_H= (42,  148,  65)
BTN_BLUE   = (30,   82, 152)
BTN_BLUE_H = (46,  112, 192)
SEC_CLR    = (85,   95, 118)

LABEL_COLORS = {
    'Description': (220,  50,  50),
    'Unit':        ( 30, 160,  60),
    'Price':       ( 30, 100, 210),
}

ROW_FILLS = [
    (0, 200, 210, 30),   # even rows — teal
    (120, 200, 255, 22), # odd rows  — blue
]
ROW_BORDERS = [
    (0, 200, 210),
    (90, 190, 255),
]


class RectSelector:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Invoice OCR — Column Selector")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.Font(None, 22)
        self.font_b = pygame.font.Font(None, 24)
        self.font_s = pygame.font.Font(None, 17)

        self.image_path = None
        self.base_surf  = None
        self.base_pil   = None
        self.base_w = self.base_h = 0

        self.rotation    = 0.0   # degrees, clockwise-positive
        self.row_pad     = 3     # inset from each row edge for OCR crop (base-img px)
        self.row_height_add = 10  # extra px added above AND below each detected center (base-img px)

        # Zoom / pan
        self.zoom   = 1.0
        self.pan_x  = 0.0
        self.pan_y  = 0.0
        self._zoom_dirty  = False
        self._zoomed_surf = None
        self.image_rect   = None

        # Middle-mouse pan
        self.panning    = False
        self.pan_start  = None
        self.pan_origin = None

        # Toolbar buttons rebuilt each frame
        self._toolbar_btns = []

        self.selected_labels = ['Description', 'Unit', 'Price']
        self._reset_state()

        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    # ── state ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.rectangles              = []
        self.current_selection_index = 0
        self.drag_rect               = None
        self.dragging                = False
        self.drag_start              = None
        self.locked_y1               = None
        self.locked_y2               = None
        # Tight centers found by detection (base-image coords)
        self._tight_rows_base  = []
        self._tight_total_base = (0.0, 0.0)
        # Displayed / OCR boundaries (derived from tight + row_height_add)
        self.row_boundaries    = []    # [(by1, by2)] base-image coords
        self.show_results      = False
        self._detecting        = False
        self._uniform_n        = 0     # >0 → use even division instead of detection bands

    # ── image / rotation ──────────────────────────────────────────────────

    def _build_base(self, pil_full):
        w, h  = pil_full.size
        scale = min(SCREEN_WIDTH / w, IMG_AREA_H / h)
        nw, nh = int(w * scale), int(h * scale)
        self.base_pil  = pil_full.resize((nw, nh), Image.Resampling.LANCZOS)
        self.base_surf = pygame.image.fromstring(
            self.base_pil.tobytes(), (nw, nh), self.base_pil.mode
        )
        self.base_w, self.base_h = nw, nh
        self._zoom_dirty = True

    def load_image(self, path):
        self.image_path = path
        self.rotation   = 0.0
        try:
            orig  = Image.open(path)
            skew  = self._auto_deskew_angle(orig)
            if abs(skew) > 0.05:
                self.rotation = round(-skew, 1)
                orig = orig.rotate(-self.rotation, expand=True,
                                   resample=Image.Resampling.BICUBIC, fillcolor='white')
                print(f"[auto-deskew] initial correction {self.rotation:+.1f}°")
            self._build_base(orig)
            self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
            self._update_image_rect()
            self._reset_state()
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Error loading image: {e}")

    def _apply_rotation(self, delta):
        if not self.image_path:
            return
        self.rotation = round(self.rotation + delta, 1)
        pil_full = Image.open(self.image_path)
        if self.rotation != 0:
            pil_full = pil_full.rotate(
                -self.rotation, expand=True,
                resample=Image.Resampling.BICUBIC, fillcolor='white',
            )
        self._build_base(pil_full)
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self._update_image_rect()
        self._reset_state()
        print(f"Rotation → {self.rotation:.1f}°")

    def _get_full(self):
        """Return (full-res rotated PIL, sx, sy)."""
        orig = Image.open(self.image_path)
        full = (orig.rotate(-self.rotation, expand=True,
                             resample=Image.Resampling.BICUBIC, fillcolor='white')
                if self.rotation != 0 else orig)
        return full, full.width / self.base_pil.width, full.height / self.base_pil.height

    # ── auto deskew ────────────────────────────────────────────────────────

    def _auto_deskew_angle(self, pil_img):
        """Detect document skew via Hough lines; returns degrees (clockwise-positive)."""
        gray = np.array(pil_img.convert('L'))
        h, w = gray.shape
        scale = min(1.0, 1500 / max(w, h, 1))
        if scale < 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_len = max(40, int(gray.shape[1] * 0.12))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                 threshold=60, minLineLength=min_len, maxLineGap=15)
        if lines is None:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1:
                continue
            ang = np.degrees(np.arctan2(dy, dx))
            if abs(ang) < 20:
                angles.append(ang)
        if not angles:
            return 0.0
        angle = float(np.median(angles))
        print(f"[auto-deskew] detected {angle:+.2f}°")
        return angle

    def _apply_auto_deskew(self):
        """Re-detect skew from current base image and apply incremental correction."""
        if self.base_pil is None:
            return
        skew = self._auto_deskew_angle(self.base_pil)
        correction = round(-skew, 1)
        if abs(correction) > 0.05:
            self._apply_rotation(correction)
            print(f"[auto-deskew] applied correction {correction:+.1f}°")
        else:
            print("[auto-deskew] document appears straight")

    # ── zoom / pan ─────────────────────────────────────────────────────────

    def _update_image_rect(self):
        w = int(self.base_w * self.zoom)
        h = int(self.base_h * self.zoom)
        self.image_rect = pygame.Rect(
            int(SCREEN_WIDTH // 2 + self.pan_x - w // 2),
            int(IMG_AREA_H   // 2 + self.pan_y - h // 2),
            w, h,
        )

    def _zoomed(self):
        if self._zoom_dirty or self._zoomed_surf is None:
            w = int(self.base_w * self.zoom)
            h = int(self.base_h * self.zoom)
            self._zoomed_surf = pygame.transform.scale(self.base_surf, (w, h))
            self._zoom_dirty  = False
        return self._zoomed_surf

    def handle_zoom(self, mouse_pos, direction):
        if mouse_pos[1] >= IMG_AREA_H:
            return
        mx, my = mouse_pos
        cx, cy = SCREEN_WIDTH // 2, IMG_AREA_H // 2
        img_x = (mx - cx - self.pan_x) / self.zoom
        img_y = (my - cy - self.pan_y) / self.zoom
        old   = self.zoom
        self.zoom = (min(self.zoom * 1.15, 10.0) if direction > 0
                     else max(self.zoom / 1.15, 0.15))
        if self.zoom != old:
            self.pan_x = mx - cx - img_x * self.zoom
            self.pan_y = my - cy - img_y * self.zoom
            self._zoom_dirty = True
            self._update_image_rect()

    # ── coordinate helpers ─────────────────────────────────────────────────

    def _screen_to_base(self, pos):
        if not self.image_rect:
            return None
        x = max(self.image_rect.left, min(pos[0], self.image_rect.right))
        y = max(self.image_rect.top,
                min(pos[1], min(self.image_rect.bottom, IMG_AREA_H - 1)))
        return (
            (x - self.image_rect.left) / self.zoom,
            (y - self.image_rect.top)  / self.zoom,
        )

    def _base_to_screen(self, bx, by):
        return (
            int(self.image_rect.left + bx * self.zoom),
            int(self.image_rect.top  + by * self.zoom),
        )

    # ── selection drag ─────────────────────────────────────────────────────

    def start_drag(self, pos):
        if pos[1] >= IMG_AREA_H:
            return
        b = self._screen_to_base(pos)
        if b is None:
            return
        self.drag_start = b
        self.dragging   = True
        bx, by = b
        self.drag_rect = ((bx, by, bx, by) if self.current_selection_index == 0
                          else (bx, self.locked_y1, bx, self.locked_y2))

    def update_drag(self, pos):
        if not self.dragging or self.drag_start is None:
            return
        b = self._screen_to_base(pos)
        if b is None:
            return
        sx, sy = self.drag_start
        cx, cy = b
        self.drag_rect = (
            (min(sx,cx), min(sy,cy), max(sx,cx), max(sy,cy))
            if self.current_selection_index == 0
            else (min(sx,cx), self.locked_y1, max(sx,cx), self.locked_y2)
        )

    def end_drag(self, pos):
        if not self.dragging or self.drag_rect is None:
            self.dragging = False; self.drag_rect = None; self.drag_start = None
            return
        x1, y1, x2, y2 = self.drag_rect
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            label = self.selected_labels[self.current_selection_index]
            if self.current_selection_index == 0:
                self.locked_y1 = y1
                self.locked_y2 = y2
            self.rectangles.append((x1, y1, x2, y2, label))
            print(f"Selected {label}: x {x1:.0f}→{x2:.0f}  y {y1:.0f}→{y2:.0f}")
            self.current_selection_index += 1
            if self.current_selection_index == len(self.selected_labels):
                threading.Thread(target=self._detect_rows_only, daemon=True).start()
        else:
            print("Selection too small, try again.")
        self.dragging = False; self.drag_rect = None; self.drag_start = None

    # ── row detection ──────────────────────────────────────────────────────

    def _morph_rows(self, gray, oy1, oy2, col_w):
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hk      = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, col_w//3), 1))
        no_rule = cv2.subtract(binary, cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk))
        dk      = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, col_w//3), 1))
        dilated = cv2.dilate(no_rule, dk, iterations=3)
        n, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        return [(oy1 + s[1], oy1 + s[1] + s[3])
                for s in stats[1:] if s[2] > col_w * 0.05 and s[3] >= 2]

    def _proj_rows(self, gray, oy1, oy2):
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h = binary.sum(axis=1).astype(float)
        if h.max() == 0:
            return []
        it = h > h.max() * 0.03
        rows, prev, s = [], False, 0
        for i, t in enumerate(it):
            if t and not prev:  s = i
            elif not t and prev: rows.append((oy1 + s, oy1 + i))
            prev = t
        if prev:
            rows.append((oy1 + s, oy2))
        return rows

    @staticmethod
    def _merge(rows, gap):
        merged = []
        for r in sorted(rows):
            if merged and r[0] <= merged[-1][1] + gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
            else:
                merged.append(list(r))
        return [tuple(r) for r in merged]

    def _detect_rows(self, full_pil, sx, sy):
        """Return tight detected row bands in full-image pixel coords."""
        best = []
        for bx1, by1, bx2, by2, label in self.rectangles:
            ox1, oy1 = int(bx1*sx), int(by1*sy)
            ox2, oy2 = int(bx2*sx), int(by2*sy)
            gray     = np.array(full_pil.crop((ox1, oy1, ox2, oy2)).convert('L'))
            gap      = max(2, int((oy2 - oy1) * 0.004))
            morph    = self._merge(self._morph_rows(gray, oy1, oy2, ox2-ox1), gap)
            proj     = self._merge(self._proj_rows(gray, oy1, oy2), gap)
            winner   = morph if len(morph) >= len(proj) else proj
            print(f"[detect] {label}: morph={len(morph)} proj={len(proj)} → {len(winner)}")
            if len(winner) > len(best):
                best = winner
        return best

    # ── row height adjustment ──────────────────────────────────────────────

    def _apply_row_height(self, tight_rows_base, total_y1, total_y2):
        """
        Expand each tight band by row_height_add pixels above and below.
        Adjacent rows are capped at their midpoint so they never overlap.
        """
        if not tight_rows_base:
            return []
        extra  = float(self.row_height_add)
        result = []
        n      = len(tight_rows_base)
        for i, (y1, y2) in enumerate(tight_rows_base):
            new_y1 = y1 - extra
            new_y2 = y2 + extra
            # Cap top edge — same rule for every row: midpoint to neighbour,
            # or image boundary for the very first row.  Do NOT snap to total_y1
            # because that absorbs any gap above the first item, making row 1 taller.
            if i == 0:
                new_y1 = max(new_y1, 0.0)
            else:
                mid    = (tight_rows_base[i-1][1] + y1) / 2.0
                new_y1 = max(new_y1, mid)
            # Cap bottom edge — same rule: midpoint or image boundary
            if i == n - 1:
                new_y2 = min(new_y2, float(self.base_h))
            else:
                mid    = (y2 + tight_rows_base[i+1][0]) / 2.0
                new_y2 = min(new_y2, mid)
            # Guarantee minimum height
            if new_y2 - new_y1 < 2:
                center = (y1 + y2) / 2.0
                new_y1, new_y2 = center - 1, center + 1
            result.append((new_y1, new_y2))
        return result

    def _nudge_boundaries(self, boundaries, x1, x2):
        """Expand each row edge outward if it lands on a dark pixel (base-image coords)."""
        if not boundaries or self.base_pil is None:
            return boundaries

        gray = np.array(self.base_pil.convert('L'))
        img_h, img_w = gray.shape
        xi1 = max(0, int(x1))
        xi2 = min(img_w, int(x2))
        if xi2 <= xi1:
            return boundaries

        DARK      = 180   # pixels below this value are considered content
        MAX_NUDGE = 8     # max extra pixels to push per edge

        result = [list(b) for b in boundaries]
        n = len(result)

        for i in range(n):
            y1, y2 = result[i]

            # Top edge: slide upward while the edge row contains dark pixels
            prev_bottom = result[i - 1][1] if i > 0 else 0.0
            for _ in range(MAX_NUDGE):
                ck = max(0, int(y1))
                if ck >= img_h:
                    break
                if gray[ck, xi1:xi2].min() >= DARK:
                    break                       # edge row is clear
                new_y1 = y1 - 1.0
                if new_y1 < prev_bottom:
                    break                       # would overlap previous row
                y1 = new_y1
            result[i][0] = y1

            # Bottom edge: slide downward while the edge row contains dark pixels
            next_top = result[i + 1][0] if i < n - 1 else float(img_h - 1)
            for _ in range(MAX_NUDGE):
                ck = min(img_h - 1, int(y2))
                if ck < 0:
                    break
                if gray[ck, xi1:xi2].min() >= DARK:
                    break                       # edge row is clear
                new_y2 = y2 + 1.0
                if new_y2 > next_top:
                    break                       # would overlap next row
                y2 = new_y2
            result[i][1] = y2

        return [tuple(r) for r in result]

    def _uniform_boundaries(self):
        """Divide _tight_total_base evenly into _uniform_n equal bands."""
        y1, y2 = self._tight_total_base
        step = (y2 - y1) / self._uniform_n
        return [(y1 + i * step, y1 + (i + 1) * step)
                for i in range(self._uniform_n)]

    def _update_row_boundaries(self):
        """Recompute self.row_boundaries from stored state."""
        if not self._tight_rows_base:
            return
        if self._uniform_n > 0:
            # Even division: detection only counted rows, bounds are split uniformly
            self.row_boundaries = self._uniform_boundaries()
            return
        boundaries = self._apply_row_height(
            self._tight_rows_base, *self._tight_total_base
        )
        if self.rectangles:
            x1 = min(r[0] for r in self.rectangles)
            x2 = max(r[2] for r in self.rectangles)
            boundaries = self._nudge_boundaries(boundaries, x1, x2)
        self.row_boundaries = boundaries

    def _shift_rows(self, delta):
        """Shift all row bands by delta base-image pixels (positive = down)."""
        if not self._tight_rows_base:
            return
        if self._uniform_n > 0:
            # Uniform mode: just shift the total bounds block and recompute
            ty1, ty2 = self._tight_total_base
            ty1 += delta
            ty2 += delta
            if ty1 < 0:
                ty2 -= ty1; ty1 = 0.0
            if ty2 > self.base_h:
                ty1 -= (ty2 - self.base_h); ty2 = float(self.base_h)
            self._tight_total_base = (ty1, ty2)
        else:
            shifted = [(y1 + delta, y2 + delta) for y1, y2 in self._tight_rows_base]
            if shifted[0][0] < 0:
                d = -shifted[0][0]
                shifted = [(y1 + d, y2 + d) for y1, y2 in shifted]
            if shifted[-1][1] > self.base_h:
                d = shifted[-1][1] - self.base_h
                shifted = [(y1 - d, y2 - d) for y1, y2 in shifted]
            self._tight_rows_base = shifted
            ty1, ty2 = self._tight_total_base
            self._tight_total_base = (ty1 + delta, ty2 + delta)
        self._update_row_boundaries()

    # ── OCR ────────────────────────────────────────────────────────────────

    def _preprocess(self, gray_pil, label):
        arr = np.array(gray_pil)
        if label == 'Description':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            arr   = clahe.apply(arr)
            arr   = cv2.fastNlMeansDenoising(arr)
        else:
            _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(arr)

    def _ocr_cell(self, crop_pil, label):
        proc = self._preprocess(crop_pil.convert('L'), label)
        if label == 'Description':
            lang, cfg = 'ell', '--psm 6'
        else:
            wl = '0123456789.,x/€$% '
            lang, cfg = 'eng', f'--psm 7 -c tessedit_char_whitelist={wl}'
        return pytesseract.image_to_string(proc, lang=lang, config=cfg).strip()

    def _ocr_from_boundaries(self, full, sx, sy):
        """Run OCR using current self.row_boundaries and return table."""
        pad_base = self.row_pad
        table    = []
        for by1, by2 in self.row_boundaries:
            ry1 = int(by1 * sy)
            ry2 = int(by2 * sy)
            pad = max(0, round(pad_base * sy))
            p1  = ry1 + pad
            p2  = ry2 - pad
            if p2 - p1 < 4:
                mid = (ry1 + ry2) // 2
                p1, p2 = mid - 2, mid + 2
            row = {}
            for bx1, _, bx2, __, label in self.rectangles:
                crop = full.crop((int(bx1*sx), p1, int(bx2*sx), p2))
                cw, ch = crop.size
                if ch < 40:
                    s    = max(2, 40 // max(ch, 1))
                    crop = crop.resize((cw*s, ch*s), Image.Resampling.LANCZOS)
                row[label] = self._ocr_cell(crop, label)
            table.append(row)
            print(row)
        return table

    def _detect_rows_only(self):
        """Count rows via detection, then divide selection evenly — no OCR."""
        self._detecting = True
        try:
            full, sx, sy = self._get_full()
            tight_orig = self._detect_rows(full, sx, sy)
            n = len(tight_orig) if tight_orig else 1
            print(f"[detect] {n} row(s) detected → dividing selection into {n} equal bands.")
            self._tight_rows_base  = ([(r[0] / sy, r[1] / sy) for r in tight_orig]
                                       if tight_orig
                                       else [(self.locked_y1, self.locked_y2)])
            self._tight_total_base = (self.locked_y1, self.locked_y2)
            self._uniform_n        = n
            self._update_row_boundaries()
        finally:
            self._detecting = False

    def _run_ocr_then_show(self):
        """Re-detect row count, divide selection evenly, then run OCR."""
        full, sx, sy = self._get_full()

        tight_orig = self._detect_rows(full, sx, sy)
        n = len(tight_orig) if tight_orig else 1
        print(f"[ocr] {n} row(s) → dividing selection into {n} equal bands.")

        self._tight_rows_base  = ([(r[0]/sy, r[1]/sy) for r in tight_orig]
                                   if tight_orig
                                   else [(self.locked_y1, self.locked_y2)])
        self._tight_total_base = (self.locked_y1, self.locked_y2)
        self._uniform_n        = n
        self._update_row_boundaries()

        table = self._ocr_from_boundaries(full, sx, sy)
        self.show_results = True
        self._open_excel_window(table)

    def _rerun_ocr(self):
        """Re-run OCR with current row_height_add / row_pad — no re-detection."""
        if not self._tight_rows_base or not self.rectangles:
            return
        full, sx, sy = self._get_full()
        table = self._ocr_from_boundaries(full, sx, sy)
        self.show_results = True
        self._open_excel_window(table)

    # ── Excel window ────────────────────────────────────────────────────────

    def _open_excel_window(self, table):
        cols = self.selected_labels
        raw  = {lbl: [r.get(lbl, '') for r in table] for lbl in cols}

        root = tk.Tk()
        root.title("Invoice OCR — Results Table")
        root.geometry("960x560")
        root.configure(bg='#f4f4f4')

        hbar = tk.Frame(root, bg='#2c3e50', pady=7)
        hbar.pack(fill='x')
        tk.Label(hbar, text=f"Invoice OCR Results  ({len(table)} rows)",
                 font=('Segoe UI', 13, 'bold'), fg='white', bg='#2c3e50'
                 ).pack(side='left', padx=12)

        tframe = tk.Frame(root, bg='white')
        tframe.pack(fill='both', expand=True, padx=10, pady=(8, 0))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('T.Treeview',
                        font=('Segoe UI', 11), rowheight=28,
                        background='white', fieldbackground='white')
        style.configure('T.Treeview.Heading',
                        font=('Segoe UI', 11, 'bold'),
                        background='#2980b9', foreground='white', relief='flat')
        style.map('T.Treeview.Heading', background=[('active', '#1a6aa0')])
        style.map('T.Treeview',         background=[('selected', '#d6eaf8')])

        tree = ttk.Treeview(tframe, columns=cols, show='headings',
                            height=18, style='T.Treeview')
        cw = {'Description': 430, 'Unit': 180, 'Price': 180}
        for col in cols:
            tree.column(col, width=cw.get(col, 200), anchor='w', minwidth=80)
        for i, row in enumerate(table):
            tree.insert('', 'end', values=tuple(row.get(c, '') for c in cols),
                        tags=('even' if i % 2 == 0 else 'odd',))
        tree.tag_configure('even', background='#f0f7ff')
        tree.tag_configure('odd',  background='white')

        vsb = ttk.Scrollbar(tframe, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        status = tk.StringVar(
            value="Click column header or button to copy column.  "
                  "Double-click a cell to copy it.")
        bframe = tk.Frame(root, bg='#f4f4f4', pady=8)
        bframe.pack(fill='x')

        btn_c = {'Description': '#c0392b', 'Unit': '#1e8449', 'Price': '#1a5276'}

        def make_copy(lbl):
            def _f():
                pyperclip.copy('\n'.join(raw[lbl]))
                status.set(f"Copied '{lbl}' column ({len(raw[lbl])} rows).")
            return _f

        tk.Label(bframe, text="Copy column →",
                 font=('Segoe UI', 10), bg='#f4f4f4').pack(side='left', padx=(10, 4))
        for col in cols:
            fn = make_copy(col)
            tree.heading(col, text=col, anchor='w', command=fn)
            tk.Button(bframe, text=col, command=fn,
                      font=('Segoe UI', 10, 'bold'),
                      fg='white', bg=btn_c.get(col, '#555'),
                      activeforeground='white', activebackground='#444',
                      relief='flat', padx=14, pady=4, cursor='hand2', bd=0
                      ).pack(side='left', padx=4)

        tk.Label(bframe, textvariable=status,
                 font=('Segoe UI', 9, 'italic'), fg='#555', bg='#f4f4f4'
                 ).pack(side='left', padx=10)

        def on_dbl(event):
            item = tree.focus()
            col  = tree.identify_column(event.x)
            idx  = int(col.replace('#', '')) - 1
            if item and 0 <= idx < len(cols):
                val = str(tree.item(item)['values'][idx])
                pyperclip.copy(val)
                status.set(f'Copied cell: "{val}"')

        tree.bind('<Double-1>', on_dbl)

        # ── JSON export & matcher launch ──────────────────────────────────
        def export_json():
            path = filedialog.asksaveasfilename(
                title="Export OCR JSON",
                defaultextension='.json',
                filetypes=[('JSON', '*.json'), ('All', '*.*')],
                initialfile='invoice_ocr.json',
                parent=root,
            )
            if not path:
                return
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(table, f, ensure_ascii=False, indent=2)
            status.set(f"Exported {len(table)} rows → {os.path.basename(path)}")

        def send_to_matcher():
            tmp = tempfile.mktemp(suffix='.json', prefix='ocr_invoice_')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(table, f, ensure_ascii=False, indent=2)
            matcher = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'invoice_matching.py')
            subprocess.Popen([sys.executable, matcher, '--json', tmp])
            status.set(f"Launched matcher with {len(table)} row(s).")

        xframe = tk.Frame(root, bg='#f4f4f4', pady=4)
        xframe.pack(fill='x')
        tk.Button(xframe, text="Export JSON",
                  command=export_json,
                  font=('Segoe UI', 10, 'bold'),
                  fg='white', bg='#7d3c98',
                  activeforeground='white', activebackground='#6c3483',
                  relief='flat', padx=14, pady=4, cursor='hand2', bd=0,
                  ).pack(side='left', padx=(10, 4))
        tk.Button(xframe, text="Send to Matcher",
                  command=send_to_matcher,
                  font=('Segoe UI', 10, 'bold'),
                  fg='white', bg='#1a6aa0',
                  activeforeground='white', activebackground='#154f78',
                  relief='flat', padx=14, pady=4, cursor='hand2', bd=0,
                  ).pack(side='left', padx=4)
        tk.Label(xframe,
                 text="Export JSON → load in Matcher  |  Send to Matcher → opens Matcher directly",
                 font=('Segoe UI', 9, 'italic'), fg='#888', bg='#f4f4f4',
                 ).pack(side='left', padx=10)

        root.mainloop()

    # ── toolbar ────────────────────────────────────────────────────────────

    def _draw_btn(self, label, rect, hovered=False, green=False, blue=False):
        if green:
            bg  = BTN_GREEN_H if hovered else BTN_GREEN
            bdr = (50, 160, 72) if hovered else (38, 130, 55)
        elif blue:
            bg  = BTN_BLUE_H if hovered else BTN_BLUE
            bdr = (65, 135, 205) if hovered else (45, 100, 172)
        else:
            bg  = BTN_HOVER if hovered else BTN_NORMAL
            bdr = (82, 92, 112) if hovered else (60, 68, 85)
        pygame.draw.rect(self.screen, bg,  rect, border_radius=5)
        pygame.draw.rect(self.screen, bdr, rect, 1, border_radius=5)
        s = self.font.render(label, True, BTN_TEXT)
        self.screen.blit(s, (rect.x + (rect.w - s.get_width())  // 2,
                              rect.y + (rect.h - s.get_height()) // 2))

    def _draw_toolbar(self, mouse_pos):
        ty = SCREEN_HEIGHT - TOOLBAR_H
        pygame.draw.rect(self.screen, TOOLBAR_BG,
                         pygame.Rect(0, ty, SCREEN_WIDTH, TOOLBAR_H))
        pygame.draw.line(self.screen, (50, 60, 80),
                         (0, ty), (SCREEN_WIDTH, ty), 2)

        self._toolbar_btns = []
        bh = 28
        by = ty + TOOLBAR_H - bh - 6   # buttons near bottom
        hy = ty + 5                      # section-header row

        x  = 10
        mx, my = mouse_pos

        def btn(label, action, w=40, green=False, blue=False):
            nonlocal x
            r = pygame.Rect(x, by, w, bh)
            self._draw_btn(label, r, r.collidepoint(mx, my),
                           green=green, blue=blue)
            self._toolbar_btns.append((r, action))
            x += w + 4

        def val(text, color):
            nonlocal x
            s = self.font_b.render(text, True, color)
            self.screen.blit(s, (x, by + (bh - s.get_height()) // 2))
            x += s.get_width() + 8

        def sec(label):
            s = self.font_s.render(label, True, SEC_CLR)
            self.screen.blit(s, (x, hy))

        def sep():
            nonlocal x
            x += 8
            pygame.draw.line(self.screen, (50, 60, 80),
                             (x, ty + 4), (x, ty + TOOLBAR_H - 4))
            x += 10

        # ── ROTATION ─────────────────────────────────────────────────────
        sec("ROTATION")
        for t, d in [("-5°", -5), ("-1°", -1), ("-0.1°", -0.1)]:
            btn(t, ('rotate', d), w=40)
        val(f"{self.rotation:+.1f}°", YELLOW)
        for t, d in [("+0.1°", 0.1), ("+1°", 1), ("+5°", 5)]:
            btn(t, ('rotate', d), w=40)
        btn("Auto", ('auto_deskew', None), w=46, blue=True)

        sep()

        # ── ROW H ────────────────────────────────────────────────────────
        sec("ROW H")
        btn("−", ('row_h', -2), w=26)
        val(f"{self.row_height_add:+d}px", CYAN)
        btn("+", ('row_h', +2), w=26)

        sep()

        # ── GAP ──────────────────────────────────────────────────────────
        sec("GAP")
        btn("−", ('pad', -1), w=26)
        val(f"{self.row_pad}px", (200, 200, 100))
        btn("+", ('pad', +1), w=26)

        sep()

        all_selected = self.current_selection_index >= len(self.selected_labels)

        if all_selected:
            sep()
            if self._detecting:
                # ── Busy indicator ────────────────────────────────────────
                sec("OCR")
                ds = self.font_b.render("Detecting…", True, YELLOW)
                self.screen.blit(ds, (x, by + (bh - ds.get_height()) // 2))
                x += ds.get_width() + 8

            elif self._tight_rows_base:
                # ── Row count ─────────────────────────────────────────────
                sec("N ROWS")
                btn("−", ('uniform_n', -1), w=26)
                val(f"{self._uniform_n}", (220, 180, 100))
                btn("+", ('uniform_n', +1), w=26)

                sep()

                # ── Shift block up / down ─────────────────────────────────
                sec("SHIFT")
                btn("↑↑", ('shift_rows', -5), w=30)
                btn("↑",  ('shift_rows', -1), w=26)
                btn("↓",  ('shift_rows', +1), w=26)
                btn("↓↓", ('shift_rows', +5), w=30)

                sep()

                # ── Run OCR ───────────────────────────────────────────────
                sec("OCR")
                btn("Run OCR", ('rerun', None), w=76, green=True)

        # ── Zoom (right-aligned) ──────────────────────────────────────────
        zs = self.font_b.render(f"Zoom {self.zoom:.1f}×", True, SEC_CLR)
        self.screen.blit(zs, (SCREEN_WIDTH - zs.get_width() - 12,
                               by + (bh - zs.get_height()) // 2))

    # ── draw ───────────────────────────────────────────────────────────────

    def draw(self, mouse_pos):
        self.screen.fill(WHITE)

        if self.base_surf and self.image_rect:
            self.screen.blit(self._zoomed(), self.image_rect.topleft)

        if self.image_rect:
            # Yellow locked-height guide lines
            if self.locked_y1 is not None:
                for by in (self.locked_y1, self.locked_y2):
                    ay = int(self.image_rect.top + by * self.zoom)
                    pygame.draw.line(self.screen, YELLOW,
                                     (self.image_rect.left, ay),
                                     (self.image_rect.right, ay), 1)

            # Row boxes — alternating colors, numbered
            if self.row_boundaries and self.rectangles:
                sx1 = int(self.image_rect.left + min(r[0] for r in self.rectangles)*self.zoom)
                sx2 = int(self.image_rect.left + max(r[2] for r in self.rectangles)*self.zoom)
                pad_s = self.row_pad * self.zoom
                w_box = max(1, sx2 - sx1)

                for i, (by1, by2) in enumerate(self.row_boundaries):
                    ay1 = int(self.image_rect.top + by1 * self.zoom + pad_s)
                    ay2 = int(self.image_rect.top + by2 * self.zoom - pad_s)
                    if ay2 <= ay1:
                        continue
                    h_box = ay2 - ay1
                    fill = pygame.Surface((w_box, h_box), pygame.SRCALPHA)
                    fill.fill(ROW_FILLS[i % 2])
                    self.screen.blit(fill, (sx1, ay1))
                    bc = ROW_BORDERS[i % 2]
                    pygame.draw.line(self.screen, bc, (sx1, ay1), (sx2, ay1), 2)
                    pygame.draw.line(self.screen, bc, (sx1, ay2), (sx2, ay2), 2)
                    pygame.draw.line(self.screen, bc, (sx1, ay1), (sx1, ay2), 1)
                    pygame.draw.line(self.screen, bc, (sx2, ay1), (sx2, ay2), 1)
                    # Row number badge
                    num_s = self.font_s.render(str(i + 1), True, bc)
                    bw, bh2 = num_s.get_width() + 6, num_s.get_height() + 2
                    badge = pygame.Surface((bw, bh2), pygame.SRCALPHA)
                    badge.fill((0, 0, 0, 150))
                    self.screen.blit(badge, (sx1 + 2, ay1 + 1))
                    self.screen.blit(num_s, (sx1 + 5, ay1 + 2))

            # Column rectangles — filled badge label, thicker border
            for bx1, by1, bx2, by2, label in self.rectangles:
                ax1, ay1 = self._base_to_screen(bx1, by1)
                ax2, ay2 = self._base_to_screen(bx2, by2)
                c = LABEL_COLORS.get(label, (128, 128, 128))
                # Semi-transparent fill
                col_fill = pygame.Surface((ax2 - ax1, ay2 - ay1), pygame.SRCALPHA)
                col_fill.fill((*c, 18))
                self.screen.blit(col_fill, (ax1, ay1))
                # 2px border
                pygame.draw.rect(self.screen, c,
                                 pygame.Rect(ax1, ay1, ax2 - ax1, ay2 - ay1), 2,
                                 border_radius=2)
                # Label badge above the column box
                lbl_s  = self.font_b.render(label, True, (255, 255, 255))
                bw     = lbl_s.get_width() + 10
                bh2    = lbl_s.get_height() + 4
                badge_y = max(0, ay1 - bh2)
                badge   = pygame.Surface((bw, bh2), pygame.SRCALPHA)
                badge.fill((*c, 210))
                pygame.draw.rect(badge, (255, 255, 255, 60),
                                 pygame.Rect(0, 0, bw, bh2), 1, border_radius=3)
                self.screen.blit(badge, (ax1, badge_y))
                self.screen.blit(lbl_s, (ax1 + 5, badge_y + 2))

            # Live drag preview
            if self.drag_rect:
                bx1, by1, bx2, by2 = self.drag_rect
                ax1, ay1 = self._base_to_screen(bx1, by1)
                ax2, ay2 = self._base_to_screen(bx2, by2)
                pygame.draw.rect(self.screen, BLACK,
                                 pygame.Rect(ax1, ay1, ax2-ax1, ay2-ay1), 2)

        # Instructions
        idx = self.current_selection_index
        if idx == 0:
            lines = ["Step 1/3 — Drag a rectangle around the 'Description' column.",
                     "This sets the row height for all selections."]
        elif idx < 3:
            lines = [f"Step {idx+1}/3 — Drag LEFT → RIGHT for "
                     f"'{self.selected_labels[idx]}' column.",
                     "Height is locked (yellow lines)."]
        elif self._detecting:
            lines = ["Detecting rows… please wait."]
        elif not self._tight_rows_base:
            lines = ["All columns selected — detecting rows…"]
        else:
            n = len(self.row_boundaries)
            lines = [f"{n} row(s) detected.  Fine-tune with Row H / Rows ↑↓, then press Run OCR.",
                     "Cyan lines = actual OCR crop after Gap inset.  Re-run = OCR only, no re-detect."]

        y = 10
        for line in lines:
            s  = self.font.render(line, True, BLACK)
            bg = pygame.Rect(8, y - 2, s.get_width() + 6, s.get_height() + 4)
            pygame.draw.rect(self.screen, LIGHT_GRAY, bg)
            self.screen.blit(s, (11, y))
            y += 24

        self._draw_toolbar(mouse_pos)
        pygame.display.flip()

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self):
        root = Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select Invoice Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        root.destroy()

        if not image_path:
            print("No image selected. Exiting.")
            return

        self.load_image(image_path)

        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEWHEEL:
                    self.handle_zoom(mouse_pos, event.y)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        handled = False
                        for rect, action in self._toolbar_btns:
                            if rect.collidepoint(event.pos):
                                kind, val = action
                                if kind == 'rotate':
                                    self._apply_rotation(val)
                                elif kind == 'auto_deskew':
                                    self._apply_auto_deskew()
                                elif kind == 'row_h':
                                    self.row_height_add += val
                                    self._update_row_boundaries()
                                elif kind == 'pad':
                                    self.row_pad = max(0, self.row_pad + val)
                                elif kind == 'shift_rows':
                                    self._shift_rows(val)
                                elif kind == 'uniform_n':
                                    self._uniform_n = max(1, self._uniform_n + val)
                                    self._update_row_boundaries()
                                elif kind == 'run_ocr':
                                    threading.Thread(
                                        target=self._run_ocr_then_show, daemon=True
                                    ).start()
                                elif kind == 'rerun':
                                    threading.Thread(
                                        target=self._rerun_ocr, daemon=True
                                    ).start()
                                handled = True
                                break
                        if not handled:
                            if self.current_selection_index < len(self.selected_labels):
                                self.start_drag(event.pos)
                    elif event.button == 2:
                        self.panning    = True
                        self.pan_start  = event.pos
                        self.pan_origin = (self.pan_x, self.pan_y)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.dragging:
                        self.end_drag(event.pos)
                    elif event.button == 2:
                        self.panning = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.update_drag(event.pos)
                    if self.panning and self.pan_start:
                        dx = event.pos[0] - self.pan_start[0]
                        dy = event.pos[1] - self.pan_start[1]
                        self.pan_x = self.pan_origin[0] + dx
                        self.pan_y = self.pan_origin[1] + dy
                        self._update_image_rect()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self._reset_state()
                    elif event.key == pygame.K_LEFTBRACKET:
                        mods = pygame.key.get_mods()
                        self._apply_rotation(-5.0 if mods & pygame.KMOD_SHIFT else -1.0)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        mods = pygame.key.get_mods()
                        self._apply_rotation(5.0 if mods & pygame.KMOD_SHIFT else 1.0)
                    elif event.key == pygame.K_COMMA:
                        self._apply_rotation(-0.1)
                    elif event.key == pygame.K_PERIOD:
                        self._apply_rotation(0.1)

            self.draw(mouse_pos)
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = RectSelector()
    app.run()
