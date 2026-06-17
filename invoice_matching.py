#!/usr/bin/env python3
"""
invoice_matching.py
Match OCR-extracted Greek invoice descriptions against an ERP product list.

Pipeline per item:
  1. OCR error correction  (digit-for-letter substitutions, garbage tokens)
  2. Strip trailing origin / place-name tokens
  3. First-token pre-filter  (narrows ERP candidates quickly)
  4. rapidfuzz token_set_ratio  (handles extra words, reordering)
  5. Score-based status: ACCEPTED ≥80 | UNCERTAIN 55-79 | NO_MATCH <55

Dependencies:
    pip install rapidfuzz

Usage:
    python invoice_matching.py
    or import and call match_all() / MatcherApp from invoice_ocr.py
"""

import os
import re
import json
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass

# ── Optional rapidfuzz ─────────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

# ── Thresholds ─────────────────────────────────────────────────────────────
ACCEPT    = 80   # score ≥ this  →  ACCEPTED  (green)
UNCERTAIN = 55   # score ≥ this  →  UNCERTAIN (yellow)
           #    score  < this  →  NO_MATCH  (red)

# ── Built-in ERP list ──────────────────────────────────────────────────────
DEFAULT_ERP = [
    "ΑΓΓΟΥΡΙΑ ΦΡΕΣΚΑ", "ΑΝΙΘΟ ΦΡΕΣΚΟ", "ΒΑΣΙΛΙΚΟΣ ΠΛΑΤΥΦ.ΜΑΤΣΟ",
    "ΓΛΥΚΟΠΑΤΑΤΑ ΚΟΚΚΙΝΗ /ΑΣΠΡΗ ΦΡΕΣΚΙΑ", "ΔΥΟΣΜΟΣ-ΜΕΝΤΑ ΜΑΝΑΒΙΚΗ",
    "ΚΑΡΟΤΑ ΦΡΕΣΚΑ", "ΚΟΛΙΑΝΔΡΟΣ(ΜΑΝΑΒΙΚΗ)", "ΚΟΛΟΚΥΘΑΚΙΑ ΦΡΕΣΚΑ(ΜΙΚΡΑ)",
    "ΚΟΛΟΚΥΘΙΑ ΦΡΕΣΚΑ", "ΚΟΥΝΟΥΠΙΔΙ ΦΡΕΣΚΟ ΜΑΝΑΒ", "ΚΡΕΜΜΥΔΑΚΙΑ ΦΡΕΣΚΑ",
    "ΚΡΕΜΜΥΔΙΑ ΞΕΡΑ", "ΛΑΙΜ ΒΡΑΖΙΛΙΑΣ", "ΛΑΧΑΝΟ ΚΟΚΚΙΝΟ ΦΡΕΣΚΟ",
    "ΛΑΧΑΝΟ ΦΡΕΣΚΟ", "ΛΕΜΟΝΙΑ", "ΜΑΙΝΤΑΝΟΣ", "ΜΑΝΙΤΑΡΙΑ ΑΣΠΡΑ ΦΡΕΣΚΑ",
    "ΜΑΡΑΘΟ ΦΡΕΣΚΟ", "ΜΑΡΟΥΛΙΑ ΦΡΕΣΚΑ (ΤΕΜ)", "ΜΑΡΟΥΛΙΑ ΦΡΕΣΚΑ ΜΕ ΤΟ ΚΙΛΟ",
    "ΜΕΛΙΤΖΑΝΕΣ ΦΡΕΣΚΕΣ", "ΠΑΤΑΤΑ ΦΡΕΣΚΙΑ ΚΟΜΜΕΝΗ", "ΠΑΤΑΤΕΣ ΒΑΒΥ(μαναβικη)",
    "ΠΑΤΑΤΕΣ ΦΡΕΣΚΙΕΣ", "ΠΑΤΖΑΡΙ ΦΡΕΣΚΟ", "ΠΙΠΕΡΙΑ ΞΑΝΘΗ ΕΓΧΩΡΙΑ",
    "ΠΙΠΕΡΙΕΣ ΕΓΧΡΩΜΕΣ", "ΠΙΠΕΡΙΕΣ ΚΕΡΑΤΟ", "ΠΙΠΕΡΙΕΣ ΚΙΤΡΙΝΕΣ",
    "ΠΙΠΕΡΙΕΣ ΚΟΚΚΙΝΕΣ", "ΠΙΠΕΡΙΕΣ ΠΟΡΤΟΚΑΛΙ", "ΠΙΠΕΡΙΕΣ ΠΡΑΣΙΝΕΣ",
    "ΠΙΠΕΡΙΕΣ ΦΛΩΡΙΝΗΣ ΦΡΕΣΚΕΣ", "ΠΡΑΣΣΕΣ", "ΡΑΠΑΝΑΚΙΑ",
    "ΡΟΚΑ (ΜΑΝΑΒΙΚΗ) ΣΕ ΤΕΜ", "ΡΟΚΑ ΒΑΒΥ 125ΓΡ", "ΡΟΚΑ ΒΑΒΥ ΜΕ ΚΙΛΟ",
    "ΡΟΚΑ ΜΑΤΣΟ", "ΣΑΛΑΤΑ ΑΙΣΜΠΕΡΓΚ", "ΣΑΛΑΤΑ ΑΙΣΜΠΕΡΚ ΣΕ ΤΕΜΑΧΙΑ",
    "ΣΑΛΑΤΑ ΓΑΛΛΙΚΗ", "ΣΑΛΑΤΑ ΛΟΛΑ ΜΠΙΟΝΤΟ- ΛΟΛΛΟ ΡΟΣΣΟ ΚΟΚΚΙΝΟ",
    "ΣΑΛΑΤΑ ΛΟΛΑ ΡΟΣΟ-ΛΟΛΟ ΡΟΣΣΟ ΓΚΡΗΝΟ (ΠΡΑΣΙΝΗ)",
    "ΣΑΛΑΤΑ ΠΡΑΣΙΝΗ ΣΓΟΥΡΗ ΣΕ ΚΙΛΑ", "ΣΑΛΑΤΑ ΠΡΑΣΙΝΗ ΣΓΟΥΡΗ ΣΕ ΤΕΜ",
    "ΣΑΛΑΤΑ ΡΑΝΤΙΤΣΙΟ", "ΣΕΛΕΡΙ (ΜΑΝΑΒΙΚΟ)", "ΣΕΛΙΝΟ/ΣΕΛΙΝΟΡΙΖΑ",
    "ΣΚΟΡΔΑ ΞΕΡΑ ΣΕ ΚΙΛΑ", "ΣΚΟΡΔΑ ΞΕΡΑ ΣΕ ΤΕΜ",
    "ΣΠΑΝΑΚΙ ΒΑΒΥ ΦΡΕΣΚΟ ΣΕ ΤΕΜ",
    "ΤΟΜΑΤΕΣ Α' ΦΡΕΣΚΕΣ(ΠΑΝΩ ΑΠΟ 1 ΕΥΡΩ)", "ΤΟΜΑΤΙΝΙΑ ΦΡΕΣΚΑ",
    "ΦΙΝΟΚΙΟ", "ΧΟΡΤΑ ΦΡΕΣΚΑ",
    "ΑΒΟΚΑΝΤΟ", "ΑΚΤΙΝΙΔΙΑ ΦΡΕΣΚΑ", "ΑΝΑΝΑΣ ΦΡΕΣΚΟΣ", "ΑΧΛΑΔΙΑ",
    "ΒΑΝΙΛΙΕΣ ΦΡΟΥΤΟ", "ΒΕΡΙΚΟΚΑ",
    "ΓΚΡΕΙΠ ΦΡΟΥΤ   (ΚΙΤΡΙΝΟ)", "ΓΚΡΕΙΠ ΦΡΟΥΤ  (ΚΟΚΚΙΝΟ)",
    "ΚΑΡΠΟΥΖΙΑ ΕΝΧΩΡΙΑ", "ΚΑΡΠΟΥΖΙΑ ΚΙΤΡΙΝΑ ΕΝΧΩΡΙΑ",
    "ΚΕΡΑΣΙΑ ΦΡΕΣΚΑ", "ΜΑΝΓΚΟ ΦΡΟΥΤΟ", "ΜΗΛΑ ΣΜΙΘ", "ΜΗΛΑ ΣΤΑΡΚΙΝ",
    "ΜΠΑΝΑΝΕΣ ΕΙΣΑΓΩΓΗΣ", "ΝΕΚΤΑΡΙΝΙΑ", "ΠΑΣΣΙΟΝ ΦΡΟΥΤ (ΜΑΝΑΒΙΚΗ)",
    "ΠΕΠΟΝΙΑ", "ΠΟΡΤΟΚΑΛΙΑ ΦΡΕΣΚΑ", "ΡΟΔΑΚΙΝΑ ΦΡΕΣΚΑ",
    "ΣΤΑΦΥΛΙΑ ΕΛΛ.", "ΦΡΑΟΥΛΕΣ ΚΤΨ", "ΦΡΑΟΥΛΕΣ ΦΡΕΣΚΕΣ", "ΦΥΣΑΛΙΔΑ ΦΡΟΥΤΟ",
]

ERP_FILE = os.path.join(os.path.dirname(__file__), 'erp_items.txt')

# ── Origin / place-name tokens to strip from the end of descriptions ───────
# These are Greek city / region / country names (genitive) and import markers
# that appear on invoices but not in ERP names.
ORIGINS = {
    # Greek regions
    'ΑΤΤΙΚΗΣ','ΒΟΙΩΤΙΑΣ','ΘΕΣΣΑΛΙΑΣ','ΜΑΚΕΔΟΝΙΑΣ','ΗΠΕΙΡΟΥ',
    'ΠΕΛΟΠΟΝΝΗΣΟΥ','ΚΡΗΤΗΣ','ΕΥΒΟΙΑΣ','ΘΡΑΚΗΣ',
    # Greek cities
    'ΠΡΕΒΕΖΑΣ','ΠΡΕΒΕΖΗΣ','ΚΑΣΤΟΡΙΑΣ','ΑΡΤΑΣ','ΛΑΡΙΣΗΣ','ΛΑΡΙΣΑΣ',
    'ΒΟΛΟΥ','ΒΟΛΟΣ','ΚΟΡΙΝΘΟΥ','ΜΕΓΑΡΩΝ','ΒΕΡΟΙΑΣ','ΦΛΩΡΙΝΗΣ',
    'ΙΩΑΝΝΙΝΩΝ','ΤΡΙΚΑΛΩΝ','ΘΕΣΣΑΛΟΝΙΚΗΣ','ΘΕΣ/ΝΙΚΗΣ','ΘΕΣ.',
    'ΧΑΝΙΩΝ','ΗΡΑΚΛΕΙΟΥ','ΣΕΡΡΩΝ','ΔΡΑΜΑΣ','ΚΑΒΑΛΑΣ','ΞΑΝΘΗΣ',
    'ΚΙΛΚΙΣ','ΠΕΛΛΑΣ','ΗΜΑΘΙΑΣ','ΠΙΕΡΙΑΣ','ΧΑΛΚΙΔΙΚΗΣ','ΚΟΖΑΝΗΣ',
    'ΑΧΑΙΑΣ','ΗΛΕΙΑΣ','ΚΟΡΙΝΘΙΑΣ','ΑΡΓΟΛΙΔΑΣ','ΑΡΚΑΔΙΑΣ',
    'ΛΑΚΩΝΙΑΣ','ΜΕΣΣΗΝΙΑΣ','ΦΘΙΩΤΙΔΑΣ','ΦΩΚΙΔΑΣ','ΧΑΛΚΙΔΑΣ','ΛΑΜΙΑΣ',
    # Foreign countries
    'ΠΟΛΩΝΙΑΣ','ΒΡΑΖΙΛΙΑΣ','ΙΣΠΑΝΙΑΣ','ΙΤΑΛΙΑΣ','ΓΑΛΛΙΑΣ',
    'ΟΛΛΑΝΔΙΑΣ','ΙΣΡΑΗΛ','ΑΙΓΥΠΤΟΥ','ΤΟΥΡΚΙΑΣ','ΜΑΡΟΚΟΥ',
    'ΚΙΝΑΣ','ΙΝΔΙΑΣ','ΧΙΛΗΣ','ΜΕΞΙΚΟΥ','ΠΕΡΟΥΒΙΑΣ','ΕΚΟΥΑΔΟΡ',
    # Import / Costa Rica abbreviations
    'ΕΙΣ.','ΕΙΣ','ΕΙΣΑΓΩΓΗΣ','Κ.','ΡΙΚΑ','ΚΟΣΤΑΡΙΚΑ',
    # Saint-name place qualifiers
    'ΑΓ.','ΓΕΩΡΓΙΟΣ','ΓΕΩΡΓΙΟΥ','ΝΙΚΟΛΑΟΣ','ΠΑΡΑΣΚΕΥΗ',
    # Pure OCR garbage tokens
    'ΝΙΞΞΟΞΣ','ΝΙΞΞ',
}

# ── Common digit-for-Greek-letter OCR substitutions ────────────────────────
_OCR_FIXES = [
    (re.compile(r'\b6([ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ])', re.I), r'Γ\1'),  # 6 → Γ
    (re.compile(r'\b8([ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ])', re.I), r'Β\1'),  # 8 → Β
    (re.compile(r'\bΝΙΞΞΟΞΣ\b|\bΝΙΞΞ\b'), ''),                        # garbage
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure matching logic  (no UI, importable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    original:  str
    unit:      str  = ''
    price:     str  = ''
    cleaned:   str  = ''
    erp_match: str  = ''
    score:     float = 0.0
    status:    str  = 'PENDING'   # ACCEPTED | UNCERTAIN | NO_MATCH | MANUAL


def _fix_ocr(text: str) -> str:
    for pat, rep in _OCR_FIXES:
        text = pat.sub(rep, text)
    return text


def _strip_origins(tokens: list) -> list:
    """Remove trailing tokens that are known origin / place names."""
    out      = list(tokens)
    max_strip = max(0, len(out) - 1)   # always keep at least the first token
    removed   = 0
    while removed < max_strip and out and out[-1].upper() in ORIGINS:
        out.pop()
        removed += 1
    return out


def clean(desc: str) -> str:
    """
    Full cleaning pipeline:
      uppercase → fix OCR errors → split → strip trailing origins → rejoin
    """
    text   = _fix_ocr(desc.upper().strip())
    tokens = [t for t in text.split() if t]
    tokens = _strip_origins(tokens)
    return ' '.join(t for t in tokens if t)


def _score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _RAPIDFUZZ:
        # token_set_ratio ignores extra words and handles reordering
        return fuzz.token_set_ratio(a, b)
    # Fallback: character-level Jaccard (no extra dependency)
    sa, sb = set(a), set(b)
    union  = sa | sb
    return 100.0 * len(sa & sb) / len(union) if union else 0.0


def _candidates(cleaned: str, erp_list: list) -> list:
    """Pre-filter ERP list to items that share the first token."""
    if not cleaned:
        return erp_list
    first = cleaned.split()[0]
    hits  = [e for e in erp_list
             if e.upper().startswith(first)
             or (len(first) >= 4 and first in e.upper())]
    return hits or erp_list


def match_one(desc: str, erp_list: list,
              unit: str = '', price: str = '') -> MatchResult:
    c = clean(desc)
    if not c or not erp_list:
        return MatchResult(original=desc, unit=unit, price=price,
                           cleaned=c, status='NO_MATCH')

    pool       = _candidates(c, erp_list)
    best, bscore = '', 0.0
    for e in pool:
        s = _score(c, e.upper())
        if s > bscore:
            bscore, best = s, e

    if   bscore >= ACCEPT:    status = 'ACCEPTED'
    elif bscore >= UNCERTAIN: status = 'UNCERTAIN'
    else:                     status = 'NO_MATCH'; best = ''

    return MatchResult(original=desc, unit=unit, price=price,
                       cleaned=c, erp_match=best,
                       score=bscore, status=status)


def match_all(rows: list, erp_list: list) -> list:
    """
    rows: list of dicts with keys 'Description', optionally 'Unit', 'Price'
          OR a plain list of strings.
    """
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(match_one(r.get('Description', ''), erp_list,
                                 unit=r.get('Unit', ''), price=r.get('Price', '')))
        else:
            out.append(match_one(str(r), erp_list))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ERP list persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_erp() -> list:
    if os.path.exists(ERP_FILE):
        try:
            with open(ERP_FILE, encoding='utf-8') as f:
                items = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
            if items:
                return items
        except Exception:
            pass
    return list(DEFAULT_ERP)


def save_erp(items: list):
    with open(ERP_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(items) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

_TAG_BG = {
    'ACCEPTED':  '#d4edda',
    'UNCERTAIN': '#fff3cd',
    'NO_MATCH':  '#f8d7da',
    'MANUAL':    '#d1ecf1',
    'PENDING':   '#f8f9fa',
}
_TAG_ICON = {
    'ACCEPTED': '✓', 'UNCERTAIN': '?',
    'NO_MATCH': '✗', 'MANUAL': '✎', 'PENDING': '…',
}
_COLS = ('#', 'Description (OCR)', 'Cleaned', 'ERP Match', 'Score', 'Status', 'Unit', 'Price')
_COL_W = {
    '#': 34, 'Description (OCR)': 270, 'Cleaned': 210,
    'ERP Match': 280, 'Score': 52, 'Status': 82, 'Unit': 65, 'Price': 65,
}


class MatcherApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Invoice → ERP Matcher")
        self.geometry("1280x680")
        self.configure(bg='#f0f0f0')

        self._erp:     list = load_erp()
        self._results: list = []

        self._build_ui()
        if not _RAPIDFUZZ:
            messagebox.showwarning(
                "rapidfuzz not found",
                "Install rapidfuzz for best results:\n  pip install rapidfuzz\n\n"
                "Falling back to basic character Jaccard similarity.")

    # ── layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg='#2c3e50', pady=8)
        hdr.pack(fill='x')
        tk.Label(hdr, text="Invoice → ERP Matcher",
                 font=('Segoe UI', 13, 'bold'), fg='white', bg='#2c3e50'
                 ).pack(side='left', padx=12)

        btn_bar = tk.Frame(hdr, bg='#2c3e50')
        btn_bar.pack(side='right', padx=10)
        for label, cmd, color in [
            ("Load OCR JSON",        self._load_json,       '#2980b9'),
            ("Paste Descriptions",   self._paste_dialog,    '#2980b9'),
            ("Edit ERP List",        self._edit_erp,        '#7f8c8d'),
            ("Run Match",            self._run_match,       '#27ae60'),
        ]:
            tk.Button(btn_bar, text=label, command=cmd,
                      font=('Segoe UI', 9), relief='flat',
                      bg=color, fg='white',
                      activebackground=color, padx=9, pady=4
                      ).pack(side='left', padx=3)

        # table
        mid = tk.Frame(self, bg='#f0f0f0')
        mid.pack(fill='both', expand=True, padx=10, pady=8)

        sty = ttk.Style()
        sty.theme_use('clam')
        sty.configure('M.Treeview', font=('Segoe UI', 10), rowheight=25,
                       background='white', fieldbackground='white')
        sty.configure('M.Treeview.Heading', font=('Segoe UI', 10, 'bold'),
                       background='#34495e', foreground='white', relief='flat')
        sty.map('M.Treeview', background=[('selected', '#aed6f1')])

        self._tree = ttk.Treeview(mid, columns=_COLS, show='headings',
                                   height=24, style='M.Treeview')
        for c in _COLS:
            self._tree.heading(c, text=c, anchor='w')
            self._tree.column(c, width=_COL_W.get(c, 120), anchor='w', minwidth=28)
        for tag, bg in _TAG_BG.items():
            self._tree.tag_configure(tag, background=bg)

        vsb = ttk.Scrollbar(mid, orient='vertical',   command=self._tree.yview)
        hsb = ttk.Scrollbar(mid, orient='horizontal', command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        self._tree.bind('<Double-1>', self._on_dbl)

        # status bar
        bot = tk.Frame(self, bg='#f0f0f0', pady=6)
        bot.pack(fill='x', padx=10)
        self._status = tk.StringVar(value="Load OCR data and press Run Match.")
        tk.Label(bot, textvariable=self._status,
                 font=('Segoe UI', 9), fg='#444', bg='#f0f0f0'
                 ).pack(side='left')
        for label, cmd, color in [
            ("Export CSV",      self._export_csv,     '#27ae60'),
            ("Copy Accepted",   self._copy_accepted,  '#27ae60'),
        ]:
            tk.Button(bot, text=label, command=cmd,
                      font=('Segoe UI', 9), relief='flat',
                      bg=color, fg='white', padx=9, pady=3
                      ).pack(side='right', padx=4)

    # ── actions ────────────────────────────────────────────────────────────

    def _load_json(self):
        path = filedialog.askopenfilename(
            title="Load OCR JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            self._apply_rows(data)
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _paste_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Paste Descriptions")
        dlg.geometry("620x420")
        dlg.grab_set()

        tk.Label(dlg,
                 text="Paste invoice descriptions — one per line.\n"
                      "Or paste full JSON (list of {Description, Unit, Price}).",
                 font=('Segoe UI', 10), justify='left'
                 ).pack(anchor='w', padx=12, pady=(12, 4))

        txt = ScrolledText(dlg, font=('Consolas', 10), height=18)
        txt.pack(fill='both', expand=True, padx=12)

        def _go():
            raw = txt.get('1.0', 'end').strip()
            dlg.destroy()
            if not raw:
                return
            # Try JSON first, fall back to plain lines
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = [{'Description': ln.strip()}
                        for ln in raw.splitlines() if ln.strip()]
            self._apply_rows(data)

        tk.Button(dlg, text="Run Match", command=_go,
                  bg='#27ae60', fg='white', relief='flat',
                  font=('Segoe UI', 10, 'bold'), padx=16, pady=5
                  ).pack(pady=10)
        dlg.bind('<Control-Return>', lambda e: _go())

    def _edit_erp(self):
        dlg = tk.Toplevel(self)
        dlg.title("Edit ERP List")
        dlg.geometry("540x620")
        dlg.grab_set()

        tk.Label(dlg, text="One ERP item per line:",
                 font=('Segoe UI', 10)).pack(anchor='w', padx=12, pady=(12, 4))

        txt = ScrolledText(dlg, font=('Consolas', 10), height=32)
        txt.pack(fill='both', expand=True, padx=12)
        txt.insert('1.0', '\n'.join(self._erp))

        def _save():
            items = [ln.strip() for ln in txt.get('1.0', 'end').splitlines()
                     if ln.strip()]
            self._erp = items
            save_erp(items)
            dlg.destroy()
            self._status.set(f"ERP list saved — {len(items)} items.")

        tk.Button(dlg, text="Save & Close", command=_save,
                  bg='#2980b9', fg='white', relief='flat',
                  font=('Segoe UI', 10, 'bold'), padx=14, pady=5
                  ).pack(pady=10)

    def _apply_rows(self, rows: list):
        """Store raw rows and immediately run matching."""
        self._rows   = rows
        self._results = []
        self._run_match()

    def _run_match(self):
        rows = getattr(self, '_rows', [])
        if not rows:
            messagebox.showinfo("No data", "Load OCR data first.")
            return
        self._status.set("Matching…")
        self.update_idletasks()

        def _go():
            results = match_all(rows, self._erp)
            self.after(0, lambda: self._display(results))

        threading.Thread(target=_go, daemon=True).start()

    def _display(self, results: list):
        self._results = results
        self._refresh()

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        counts = {'ACCEPTED': 0, 'UNCERTAIN': 0, 'NO_MATCH': 0, 'MANUAL': 0}

        for i, r in enumerate(self._results, 1):
            icon  = _TAG_ICON.get(r.status, '')
            score = f"{r.score:.0f}" if r.score else '—'
            self._tree.insert('', 'end', iid=str(i), tags=(r.status,),
                              values=(i, r.original, r.cleaned,
                                      r.erp_match or '—', score,
                                      f"{icon} {r.status}",
                                      r.unit, r.price))
            if r.status in counts:
                counts[r.status] += 1

        n = len(self._results)
        self._status.set(
            f"{n} items  |  "
            f"✓ Accepted: {counts['ACCEPTED']}   "
            f"? Uncertain: {counts['UNCERTAIN']}   "
            f"✗ No match: {counts['NO_MATCH']}   "
            f"✎ Manual: {counts['MANUAL']}   "
            f"   Double-click any row to override."
        )

    # ── double-click override ───────────────────────────────────────────────

    def _on_dbl(self, event):
        iid = self._tree.focus()
        if not iid:
            return
        idx = int(iid) - 1
        self._override_dialog(idx)

    def _override_dialog(self, idx: int):
        r   = self._results[idx]
        dlg = tk.Toplevel(self)
        dlg.title("Override Match")
        dlg.geometry("600x210")
        dlg.grab_set()

        tk.Label(dlg, text=f"Invoice:   {r.original}",
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=14, pady=(14, 2))
        tk.Label(dlg, text=f"Cleaned:  {r.cleaned}",
                 font=('Segoe UI', 9), fg='#555').pack(anchor='w', padx=14)
        tk.Label(dlg, text=f"Current:   {r.erp_match or '— none —'}  (score {r.score:.0f})",
                 font=('Segoe UI', 9), fg='#555').pack(anchor='w', padx=14, pady=(2, 8))

        tk.Label(dlg, text="Select correct ERP item:",
                 font=('Segoe UI', 10)).pack(anchor='w', padx=14)

        var = tk.StringVar(value=r.erp_match or '')
        cb  = ttk.Combobox(dlg, textvariable=var,
                            values=['— NO MATCH —'] + self._erp,
                            width=58, font=('Segoe UI', 10))
        cb.pack(padx=14, fill='x')
        cb.focus_set()

        def _confirm():
            chosen = var.get().strip()
            if chosen and chosen != '— NO MATCH —':
                self._results[idx].erp_match = chosen
                self._results[idx].score     = 100.0
                self._results[idx].status    = 'MANUAL'
            else:
                self._results[idx].erp_match = ''
                self._results[idx].score     = 0.0
                self._results[idx].status    = 'NO_MATCH'
            dlg.destroy()
            self._refresh()

        tk.Button(dlg, text="Confirm", command=_confirm,
                  bg='#2980b9', fg='white', relief='flat',
                  font=('Segoe UI', 10, 'bold'), padx=16, pady=5
                  ).pack(pady=10)
        dlg.bind('<Return>', lambda e: _confirm())

    # ── export ─────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile='matched_invoice.csv')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['#', 'Description', 'Cleaned', 'ERP Match',
                        'Score', 'Status', 'Unit', 'Price'])
            for i, r in enumerate(self._results, 1):
                w.writerow([i, r.original, r.cleaned, r.erp_match,
                             f"{r.score:.0f}", r.status, r.unit, r.price])
        self._status.set(f"Exported → {path}")

    def _copy_accepted(self):
        """Copy accepted+manual matches as tab-separated text (paste into Excel)."""
        lines = ['Description\tERP Match\tUnit\tPrice']
        for r in self._results:
            if r.status in ('ACCEPTED', 'MANUAL') and r.erp_match:
                lines.append(f"{r.original}\t{r.erp_match}\t{r.unit}\t{r.price}")
        self.clipboard_clear()
        self.clipboard_append('\n'.join(lines))
        self._status.set(f"Copied {len(lines) - 1} accepted rows to clipboard.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    MatcherApp().mainloop()
