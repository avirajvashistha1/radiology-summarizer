"""
Generate three professional diagrams for the Radiology Report Summarizer project:
  1. class_diagram.pdf
  2. architectural_diagram.pdf
  3. flow_diagram.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUT = Path(__file__).parent

# ─── Colour palette ────────────────────────────────────────────────────────────
C = {
    "navy":        "#1B2A4A",
    "blue":        "#2563EB",
    "sky":         "#3B82F6",
    "teal":        "#0D9488",
    "green":       "#16A34A",
    "amber":       "#D97706",
    "orange":      "#EA580C",
    "red":         "#DC2626",
    "purple":      "#7C3AED",
    "violet":      "#6D28D9",
    "slate":       "#475569",
    "light_slate": "#94A3B8",
    "white":       "#FFFFFF",
    "off_white":   "#F8FAFC",
    "bg":          "#0F172A",
    "panel":       "#1E293B",
    "panel2":      "#263147",
}


def add_box(ax, x, y, w, h, label, sublabel=None,
            fc=C["panel"], ec=C["sky"], lw=1.5, fontsize=9,
            text_color=C["white"], radius=0.02):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h * 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=C["light_slate"], zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C["sky"], lw=1.4, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=5)


def section_header(ax, x, y, w, h, title, fc=C["navy"], ec=C["sky"]):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0,rounding_size=0.015",
                         facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["white"], zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASS DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def make_class_diagram():
    fig, ax = plt.subplots(figsize=(22, 16))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Title
    ax.text(11, 15.4, "Radiology Report Summarizer — Class Diagram",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["white"])
    ax.text(11, 15.0, "Module-level functions, Pydantic models, and key constants",
            ha="center", va="center", fontsize=10, color=C["light_slate"])

    # ── Helper to draw a UML-style class box ──────────────────────────────
    def uml_box(cx, cy, w, title, title_color, header_fc,
                methods=None, attrs=None):
        line_h = 0.38
        n_methods = len(methods) if methods else 0
        n_attrs   = len(attrs)   if attrs   else 0
        total_rows = n_methods + n_attrs + (1 if n_attrs and n_methods else 0)
        body_h = max(total_rows * line_h + 0.25, 0.5)
        header_h = 0.55
        total_h = header_h + body_h

        # Header
        hdr = FancyBboxPatch((cx - w/2, cy - total_h/2 + body_h),
                             w, header_h,
                             boxstyle="round,pad=0,rounding_size=0.015",
                             facecolor=header_fc, edgecolor=C["sky"],
                             linewidth=1.8, zorder=3)
        ax.add_patch(hdr)
        ax.text(cx, cy - total_h/2 + body_h + header_h/2, title,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=title_color, zorder=4)

        # Body
        bdy = FancyBboxPatch((cx - w/2, cy - total_h/2),
                             w, body_h,
                             boxstyle="round,pad=0,rounding_size=0.01",
                             facecolor=C["panel"], edgecolor=C["sky"],
                             linewidth=1.8, zorder=3)
        ax.add_patch(bdy)

        # Divider between attrs and methods
        row_y = cy - total_h/2 + body_h - 0.12
        if attrs:
            for a in attrs:
                row_y -= line_h
                ax.text(cx - w/2 + 0.15, row_y, a, ha="left", va="center",
                        fontsize=7.5, color=C["amber"], zorder=4,
                        fontfamily="monospace")
        if attrs and methods:
            divider_y = row_y - 0.08
            ax.plot([cx - w/2 + 0.05, cx + w/2 - 0.05],
                    [divider_y, divider_y], color=C["slate"], lw=0.8, zorder=4)
            row_y = divider_y - 0.08
        if methods:
            for m in methods:
                row_y -= line_h
                ax.text(cx - w/2 + 0.15, row_y, m, ha="left", va="center",
                        fontsize=7.5, color=C["teal"], zorder=4,
                        fontfamily="monospace")

        return total_h  # return height for arrow anchoring

    # ── Data Layer ─────────────────────────────────────────────────────────
    section_header(ax, 3.5, 14.3, 6.5, 0.45,
                   "DATA PROCESSING LAYER", fc=C["teal"])

    uml_box(2.0, 12.0, 3.0, "parse_xml", C["white"], C["teal"],
            attrs=["input_dir: str", "output_path: str"],
            methods=["extract_fields(xml_path) → dict",
                     "parse_corpus(input_dir, output) → df",
                     "main()"])

    uml_box(5.5, 12.0, 3.0, "clean", C["white"], C["teal"],
            attrs=["input_path: str", "output_path: str"],
            methods=["normalize_text(text) → str",
                     "clean_reports(input, output) → df",
                     "main()"])

    uml_box(9.0, 12.0, 3.0, "split", C["white"], C["teal"],
            attrs=["RANDOM_SEED: int = 42"],
            methods=["split_data(input, output_dir) → dict",
                     "main()"])

    # ── Modeling Layer ──────────────────────────────────────────────────────
    section_header(ax, 11, 10.6, 10.5, 0.45,
                   "MODELING LAYER", fc=C["blue"])

    uml_box(3.5, 9.1, 3.2, "tokenize_dataset", C["white"], C["blue"],
            attrs=["MAX_SOURCE_LENGTH = 512",
                   "MAX_TARGET_LENGTH = 128"],
            methods=["build_tokenize_fn(tokenizer, ...) → fn",
                     "tokenize_splits(data_dir, output) → DatasetDict",
                     "main()"])

    uml_box(7.5, 9.1, 3.2, "train", C["white"], C["blue"],
            attrs=["RANDOM_SEED = 42",
                   "MAX_SOURCE_LENGTH = 512",
                   "MAX_TARGET_LENGTH = 128",
                   "NUM_BEAMS = 4"],
            methods=["set_all_seeds(seed)",
                     "build_compute_metrics(tokenizer) → fn",
                     "train(tokenized_dir, output_dir) → Trainer",
                     "main()"])

    uml_box(11.8, 9.1, 3.2, "evaluate", C["white"], C["blue"],
            attrs=["MAX_SOURCE_LENGTH = 512",
                   "MAX_TARGET_LENGTH = 128",
                   "NUM_BEAMS = 4",
                   "SAMPLE_FINDINGS: list[str]"],
            methods=["generate_summary(findings, model, ...) → str",
                     "evaluate_model(model_dir, tokenized_dir)",
                     "main()"])

    # ── Pydantic Models ─────────────────────────────────────────────────────
    section_header(ax, 17.5, 14.3, 5.5, 0.45,
                   "PYDANTIC MODELS (API)", fc=C["purple"])

    uml_box(16.0, 12.5, 2.8, "SummarizeRequest", C["white"], C["purple"],
            attrs=["findings: str  (non-empty)"])

    uml_box(19.2, 12.5, 2.8, "SummarizeResponse", C["white"], C["purple"],
            attrs=["impression: str"])

    uml_box(17.6, 11.0, 2.8, "HealthResponse", C["white"], C["purple"],
            attrs=["status: str", "model_loaded: bool"])

    # ── API Layer ───────────────────────────────────────────────────────────
    section_header(ax, 17.5, 9.8, 5.5, 0.45,
                   "API LAYER (FastAPI)", fc=C["orange"])

    uml_box(17.5, 8.0, 5.0, "app  (FastAPI)", C["white"], C["orange"],
            attrs=["_model: AutoModelForSeq2SeqLM",
                   "_tokenizer: AutoTokenizer",
                   "_device: torch.device",
                   "_model_loaded: bool",
                   "MAX_SOURCE_LENGTH = 512",
                   "MAX_TARGET_LENGTH = 128",
                   "NUM_BEAMS = 4"],
            methods=["load_model()",
                     "lifespan(app) [async ctx mgr]",
                     "POST /summarize → SummarizeResponse",
                     "GET  /health   → HealthResponse",
                     "GET  /         → dict"])

    # ── Scripts ─────────────────────────────────────────────────────────────
    section_header(ax, 3.5, 6.9, 5.0, 0.45,
                   "SCRIPTS", fc=C["amber"])

    uml_box(3.5, 5.4, 4.6, "upload_to_hub", C["white"], C["amber"],
            attrs=["model_dir: str", "hub_name: str"],
            methods=["upload_model(model_dir, hub_name, private)",
                     "main()"])

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_items = [
        (C["teal"],   "Data Layer"),
        (C["blue"],   "Modeling Layer"),
        (C["purple"], "Pydantic Models"),
        (C["orange"], "API Layer"),
        (C["amber"],  "Scripts"),
        (C["amber"],  "Constants / Attributes (orange)"),
        (C["teal"],   "Methods (teal)"),
    ]
    lx, ly = 15.0, 4.2
    ax.text(lx - 0.1, ly + 0.6, "Legend", fontsize=9, fontweight="bold",
            color=C["white"])
    for i, (col, label) in enumerate(legend_items):
        rect = FancyBboxPatch((lx - 0.1, ly - i * 0.42),
                              0.28, 0.28,
                              boxstyle="round,pad=0.02",
                              facecolor=col, edgecolor=C["white"],
                              linewidth=0.8, zorder=4)
        ax.add_patch(rect)
        ax.text(lx + 0.28, ly - i * 0.42 + 0.14, label,
                fontsize=8, color=C["light_slate"], va="center")

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "class_diagram.pdf", dpi=200,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("  [OK] class_diagram.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURAL DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def make_architectural_diagram():
    fig, ax = plt.subplots(figsize=(24, 16))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis("off")

    ax.text(12, 15.5, "Radiology Report Summarizer — Architectural Diagram",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=C["white"])
    ax.text(12, 15.1, "System components, data stores, and service boundaries",
            ha="center", va="center", fontsize=10, color=C["light_slate"])

    # ── Helper: zone rectangle ─────────────────────────────────────────────
    def zone(x, y, w, h, title, fc, ec, alpha=0.12):
        z = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1,rounding_size=0.15",
                           facecolor=fc, edgecolor=ec,
                           linewidth=2, alpha=alpha, zorder=1)
        ax.add_patch(z)
        ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=ec, alpha=0.9, zorder=2)

    # ── Helper: component box ──────────────────────────────────────────────
    def comp(cx, cy, w, h, title, icon="", fc=C["panel"], ec=C["sky"],
             fs=9, sub=""):
        b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0,rounding_size=0.07",
                           facecolor=fc, edgecolor=ec,
                           linewidth=1.8, zorder=3)
        ax.add_patch(b)
        label = f"{icon}  {title}" if icon else title
        yo = 0.12 if sub else 0
        ax.text(cx, cy + yo, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["white"], zorder=4)
        if sub:
            ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                    fontsize=7.5, color=C["light_slate"], zorder=4, style="italic")

    # ── Helper: data store (cylinder-ish) ─────────────────────────────────
    def store(cx, cy, w, h, title, fc=C["panel2"], ec=C["amber"]):
        b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0,rounding_size=0.1",
                           facecolor=fc, edgecolor=ec,
                           linewidth=1.6, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=8, fontweight="bold", color=C["amber"], zorder=4)

    # ── Helper: labelled arrow ─────────────────────────────────────────────
    def larrow(x1, y1, x2, y2, label="", color=C["sky"], lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=14),
                    zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                    fontsize=7.5, color=color, zorder=6,
                    bbox=dict(fc=C["bg"], ec="none", pad=1))

    # ═══ Zone 1 — External Inputs ══════════════════════════════════════════
    zone(0.3, 12.6, 4.2, 2.2, "External Data Source", C["green"], C["green"])
    comp(2.4, 13.8, 3.4, 0.8, "Open-i Dataset (XML)", "",
         fc="#14532d", ec=C["green"], fs=8.5, sub="~3600 radiology reports")

    # ═══ Zone 2 — Data Pipeline ════════════════════════════════════════════
    zone(0.3, 7.5, 8.8, 4.9, "Data Pipeline", C["teal"], C["teal"])
    comp(2.4, 11.4, 3.0, 0.75, "parse_xml.py", "",
         fc="#0f4c40", ec=C["teal"], fs=9, sub="XML -> CSV/Parquet")
    comp(2.4, 10.1, 3.0, 0.75, "clean.py", "",
         fc="#0f4c40", ec=C["teal"], fs=9, sub="Normalize + Deduplicate")
    comp(2.4, 8.8, 3.0, 0.75, "split.py", "",
         fc="#0f4c40", ec=C["teal"], fs=9, sub="72 / 8 / 20  split")
    store(6.5, 10.8, 2.2, 0.6, "data/processed/")
    store(6.5, 9.9, 2.2, 0.6, "data/cleaned/")
    store(6.5, 9.0, 2.2, 0.6, "train / val / test")

    # ═══ Zone 3 — Modeling Pipeline ════════════════════════════════════════
    zone(9.5, 7.5, 8.8, 4.9, "Modeling Pipeline", C["blue"], C["blue"])
    comp(11.5, 11.4, 3.2, 0.75, "tokenize_dataset.py", "",
         fc="#1e3a5f", ec=C["blue"], fs=9, sub="BART tokenizer  512/128 tok")
    comp(11.5, 10.1, 3.2, 0.75, "train.py", "",
         fc="#1e3a5f", ec=C["blue"], fs=9, sub="Seq2SeqTrainer  5 epochs")
    comp(11.5, 8.8, 3.2, 0.75, "evaluate.py", "",
         fc="#1e3a5f", ec=C["blue"], fs=9, sub="ROUGE-1/2/L on test set")
    store(15.2, 10.8, 2.2, 0.6, "data/tokenized/")
    store(15.2, 9.9, 2.2, 0.6, "model/")
    store(15.2, 9.0, 2.2, 0.6, "results/")

    # ═══ Zone 4 — Base Model ═══════════════════════════════════════════════
    zone(9.5, 12.0, 8.8, 2.7, "Pretrained Base Model", C["violet"], C["violet"])
    comp(13.9, 13.4, 5.6, 0.9, "facebook/bart-base", "",
         fc="#2e1065", ec=C["violet"], fs=10, sub="139M params  Seq2Seq Transformer")

    # ═══ Zone 5 — HF Hub ═══════════════════════════════════════════════════
    zone(18.7, 7.5, 5.0, 4.9, "HuggingFace Hub", C["amber"], C["amber"])
    comp(21.2, 11.4, 4.0, 0.75, "upload_to_hub.py", "",
         fc="#451a03", ec=C["amber"], fs=9, sub="HfApi.push_to_hub()")
    comp(21.2, 10.1, 4.0, 0.75, "HF Model Repository", "",
         fc="#451a03", ec=C["amber"], fs=9, sub="your-user/radiology-summarizer")

    # ═══ Zone 6 — Deployment ═══════════════════════════════════════════════
    zone(0.3, 1.0, 23.4, 6.2, "Deployment — Render.com (Docker)", C["orange"], C["orange"])

    comp(5.0, 5.5, 4.2, 0.8, "Docker Container", "",
         fc="#431407", ec=C["orange"], fs=9.5, sub="Python 3.11 + uvicorn")
    comp(5.0, 4.3, 4.2, 0.8, "FastAPI app.py", "",
         fc="#431407", ec=C["orange"], fs=9.5, sub="lifespan -> load_model()")
    comp(5.0, 3.1, 4.2, 0.75, "POST /summarize", "",
         fc="#5c1a08", ec="#fb923c", fs=9, sub="tokenize -> generate -> decode")
    comp(5.0, 2.3, 4.2, 0.55, "GET  /health", "",
         fc="#5c1a08", ec="#fb923c", fs=8.5)

    comp(12.0, 4.5, 4.0, 0.8, "CORS Middleware", "",
         fc="#431407", ec=C["orange"], fs=9, sub="Allow all origins")
    comp(12.0, 3.3, 4.0, 0.8, "Env Variables", "",
         fc="#431407", ec=C["orange"], fs=9,
         sub="HF_MODEL_NAME  MODEL_DIR  PORT")

    comp(19.0, 4.5, 4.0, 0.8, "Pydantic Models", "",
         fc="#431407", ec=C["orange"], fs=9,
         sub="SummarizeRequest / Response")
    comp(19.0, 3.3, 4.0, 0.8, "Health Check", "",
         fc="#431407", ec=C["orange"], fs=9, sub="HealthResponse")

    # ═══ Zone 7 — Client ═══════════════════════════════════════════════════
    comp(21.5, 1.55, 2.6, 0.7, "HTTP Client", "",
         fc=C["panel2"], ec=C["light_slate"], fs=9,
         sub="curl / browser / app")

    # ─── Arrows ────────────────────────────────────────────────────────────
    # External → parse_xml
    larrow(2.4, 13.4, 2.4, 11.78, "XML files", C["green"])
    # parse_xml → store
    larrow(3.9, 11.4, 5.4, 10.8, "CSV/Parquet", C["teal"], 1.2)
    # parse_xml → clean
    larrow(2.4, 11.03, 2.4, 10.48, "", C["teal"])
    # clean → store
    larrow(3.9, 10.1, 5.4, 9.9, "cleaned CSV", C["teal"], 1.2)
    # clean → split
    larrow(2.4, 9.73, 2.4, 9.18, "", C["teal"])
    # split → store
    larrow(3.9, 8.8, 5.4, 9.0, "splits", C["teal"], 1.2)

    # split → tokenize (cross-zone)
    larrow(9.0, 9.0, 9.9, 10.95, "CSVs", C["sky"])
    # tokenize → store
    larrow(13.1, 11.4, 14.1, 10.8, "DatasetDict", C["blue"], 1.2)
    # tokenize → train
    larrow(11.5, 11.03, 11.5, 10.48, "", C["blue"])
    # train → store
    larrow(13.1, 10.1, 14.1, 9.9, "checkpoint", C["blue"], 1.2)
    # train → evaluate
    larrow(11.5, 9.73, 11.5, 9.18, "", C["blue"])
    # evaluate → store
    larrow(13.1, 8.8, 14.1, 9.0, "metrics", C["blue"], 1.2)

    # bart-base → train
    larrow(13.9, 12.95, 12.2, 10.48, "pretrained\nweights", C["violet"])

    # model → upload
    larrow(17.4, 9.9, 19.2, 11.1, "model/", C["amber"])
    # upload → HF repo
    larrow(21.2, 11.03, 21.2, 10.48, "push_to_hub()", C["amber"])

    # model → FastAPI (deployment)
    larrow(14.5, 8.6, 6.8, 5.8, "load_model()", C["orange"], 1.3)
    # HF repo → FastAPI (deployment)
    larrow(19.2, 9.73, 6.8, 5.6, "HF_MODEL_NAME", C["amber"], 1.2)

    # FastAPI internal
    larrow(5.0, 5.1, 5.0, 4.7, "", C["orange"])
    larrow(5.0, 3.9, 5.0, 3.5, "", C["orange"])
    larrow(5.0, 2.93, 5.0, 2.58, "", C["orange"])

    # Client → FastAPI
    larrow(21.5, 1.95, 7.1, 2.7, "HTTP request", C["light_slate"])
    larrow(7.1, 2.9, 21.5, 2.1, "JSON response", C["green"])

    # Legend
    lx, ly = 21.0, 7.2
    ax.text(lx, ly, "Legend", fontsize=9, fontweight="bold", color=C["white"])
    items = [
        (C["green"],  "External Data"),
        (C["teal"],   "Data Pipeline"),
        (C["blue"],   "Modeling"),
        (C["violet"], "Base Model"),
        (C["amber"],  "Data Store / HF Hub"),
        (C["orange"], "Deployment / API"),
    ]
    for i, (col, lbl) in enumerate(items):
        r = FancyBboxPatch((lx, ly - 0.55 - i*0.46), 0.25, 0.28,
                           boxstyle="round,pad=0.02",
                           facecolor=col, edgecolor=C["white"],
                           linewidth=0.8, zorder=4)
        ax.add_patch(r)
        ax.text(lx + 0.35, ly - 0.55 - i*0.46 + 0.14, lbl,
                fontsize=8, color=C["light_slate"], va="center")

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "architectural_diagram.pdf", dpi=200,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("  [OK] architectural_diagram.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FLOW DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def make_flow_diagram():
    fig, ax = plt.subplots(figsize=(18, 26))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 26)
    ax.axis("off")

    ax.text(9, 25.5, "Radiology Report Summarizer — Flow Diagram",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["white"])
    ax.text(9, 25.1, "End-to-end data and control flow from raw data to API inference",
            ha="center", va="center", fontsize=10, color=C["light_slate"])

    # ── Helpers ────────────────────────────────────────────────────────────
    def node(cx, cy, w, h, title, fc, ec, fs=9.5, sub=""):
        b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0,rounding_size=0.1",
                           facecolor=fc, edgecolor=ec,
                           linewidth=2, zorder=3)
        ax.add_patch(b)
        yo = 0.13 if sub else 0
        ax.text(cx, cy + yo, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["white"], zorder=4)
        if sub:
            ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                    fontsize=7.5, color=C["light_slate"], zorder=4, style="italic")

    def diamond(cx, cy, hw, hh, label, fc=C["panel"], ec=C["sky"]):
        pts = [(cx, cy+hh), (cx+hw, cy), (cx, cy-hh), (cx-hw, cy)]
        poly = plt.Polygon(pts, closed=True,
                           facecolor=fc, edgecolor=ec,
                           linewidth=2, zorder=3)
        ax.add_patch(poly)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C["white"], zorder=4)

    def stadium(cx, cy, w, h, label, fc, ec, fs=9):
        b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle=f"round,pad=0,rounding_size={h/2}",
                           facecolor=fc, edgecolor=ec,
                           linewidth=2.2, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C["white"], zorder=4)

    def a(x1, y1, x2, y2, label="", color=C["sky"], lw=1.6, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   mutation_scale=14,
                                   connectionstyle=f"arc3,rad={rad}"),
                    zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.15, my, label, ha="left", va="center",
                    fontsize=8, color=color, zorder=6,
                    bbox=dict(fc=C["bg"], ec="none", pad=1))

    def phase_label(x, y, label, color):
        ax.text(x, y, label, ha="left", va="center",
                fontsize=8.5, color=color, style="italic", fontweight="bold",
                bbox=dict(fc=C["bg"], ec=color, pad=3, boxstyle="round,pad=0.2"))

    # ── Phase labels (left column) ─────────────────────────────────────────
    phase_label(0.3, 24.3, "Phase 1 — Ingest",         C["green"])
    phase_label(0.3, 22.2, "Phase 2 — Parse",          C["teal"])
    phase_label(0.3, 19.8, "Phase 3 — Clean",          C["teal"])
    phase_label(0.3, 17.7, "Phase 4 — Split",          C["teal"])
    phase_label(0.3, 15.5, "Phase 5 — Tokenize",       C["blue"])
    phase_label(0.3, 13.0, "Phase 6 — Train",          C["blue"])
    phase_label(0.3, 10.5, "Phase 7 — Evaluate",       C["blue"])
    phase_label(0.3,  8.0, "Phase 8 — Publish",        C["amber"])
    phase_label(0.3,  5.5, "Phase 9 — Serve (API)",    C["orange"])
    phase_label(0.3,  2.8, "Phase 10 — Inference",     C["orange"])

    # ── Nodes ──────────────────────────────────────────────────────────────
    # Start
    stadium(9, 24.3, 3.2, 0.65, "START", fc=C["green"], ec="#16a34a", fs=11)

    # Phase 2 — Parse
    node(9, 22.8, 5.5, 0.7, "Load XML Files", fc="#0f4c40", ec=C["teal"],
         sub="data/raw/*.xml  (Open-i corpus)")
    node(9, 21.8, 5.5, 0.7, "extract_fields(xml_path)",
         fc="#0f4c40", ec=C["teal"],
         sub="ElementTree parse → {report_id, findings, impression}")
    diamond(9, 20.8, 2.0, 0.55, "Parse OK?", fc="#0f4c40", ec=C["teal"])
    node(9, 19.8, 5.5, 0.7, "save reports.csv + .parquet",
         fc="#0f4c40", ec=C["teal"], sub="data/processed/")

    # Phase 3 — Clean
    node(9, 18.6, 5.5, 0.7, "normalize_text()",
         fc="#0f2020", ec=C["teal"],
         sub="strip whitespace · collapse spaces")
    node(9, 17.7, 5.5, 0.7, "Drop nulls + Deduplicate",
         fc="#0f2020", ec=C["teal"],
         sub="exact (findings, impression) pairs removed")
    node(9, 16.8, 5.5, 0.7, "Save cleaned CSV + Parquet",
         fc="#0f2020", ec=C["teal"], sub="data/cleaned/reports_clean.csv")

    # Phase 4 — Split
    node(9, 15.7, 5.5, 0.7, "train_test_split  80 / 20",
         fc="#0f3040", ec=C["sky"],
         sub="random_state=42")
    node(9, 14.8, 5.5, 0.7, "train_test_split  90 / 10  (on 80%)",
         fc="#0f3040", ec=C["sky"],
         sub="→ 72% train  /  8% val  /  20% test")
    node(9, 13.9, 5.5, 0.7, "Assert zero leakage  →  Save splits",
         fc="#0f3040", ec=C["sky"], sub="train.csv  val.csv  test.csv")

    # Phase 5 — Tokenize
    node(9, 12.8, 5.5, 0.7, "build_tokenize_fn(bart-base tokenizer)",
         fc="#1e3a5f", ec=C["blue"],
         sub="MAX_SOURCE=512  MAX_TARGET=128")
    node(9, 11.9, 5.5, 0.7, "dataset.map(batched=True, batch=1000)",
         fc="#1e3a5f", ec=C["blue"],
         sub="findings→input_ids  impression→labels(-100 pads)")
    node(9, 11.0, 5.5, 0.7, "Save DatasetDict to disk",
         fc="#1e3a5f", ec=C["blue"], sub="data/tokenized/")

    # Phase 6 — Train
    node(9, 9.9, 5.5, 0.7, "Load facebook/bart-base  (139M params)",
         fc="#1e1a40", ec=C["violet"],
         sub="AutoModelForSeq2SeqLM")
    node(9, 9.0, 5.5, 0.7, "Seq2SeqTrainer  (5 epochs)",
         fc="#1e1a40", ec=C["violet"],
         sub="batch=4 · grad_accum=4 · lr=3e-5 · warmup=100")
    node(9, 8.1, 5.5, 0.7, "eval per epoch  →  save best on ROUGE-1",
         fc="#1e1a40", ec=C["violet"], sub="model/")

    # Phase 7 — Evaluate
    node(9, 7.0, 5.5, 0.7, "Batch inference on test set (batch=8)",
         fc="#0d2040", ec=C["blue"],
         sub="model.generate(num_beams=4, max_length=128)")
    node(9, 6.1, 5.5, 0.7, "Compute ROUGE-1 / ROUGE-2 / ROUGE-L",
         fc="#0d2040", ec=C["blue"], sub="results/eval_metrics.json")
    diamond(9, 5.2, 2.5, 0.65, "ROUGE-1 ≥ 0.30?", fc="#0d2040", ec=C["blue"])

    # Phase 8 — Publish
    node(9, 4.1, 5.5, 0.7, "upload_model() → HF Hub",
         fc="#3b1a00", ec=C["amber"],
         sub="HfApi.push_to_hub()  model + tokenizer")

    # Phase 9 — Serve
    node(9, 3.0, 5.5, 0.7, "load_model() at FastAPI startup",
         fc="#431407", ec=C["orange"],
         sub="env: HF_MODEL_NAME or MODEL_DIR")

    # Phase 10 — Inference
    node(9, 2.0, 5.5, 0.7, "POST /summarize  { findings: str }",
         fc="#5c1a08", ec="#fb923c",
         sub="tokenize → model.generate() → decode")
    node(9, 1.0, 5.5, 0.7, "Return { impression: str }  HTTP 200",
         fc="#5c1a08", ec=C["green"], sub="or 503 if model not loaded")

    # End
    stadium(9, 0.22, 3.2, 0.55, "END", fc=C["red"], ec=C["red"], fs=11)

    # ── Skip/error node ────────────────────────────────────────────────────
    node(14.5, 20.8, 2.8, 0.6, "Log error,\nskip record",
         fc="#4a0000", ec=C["red"], fs=8.5)

    # ── Recheck node ──────────────────────────────────────────────────────
    node(14.5, 5.2, 2.8, 0.6, "Re-check\nhyperparams",
         fc="#4a0000", ec=C["red"], fs=8.5)

    # ── Arrows ─────────────────────────────────────────────────────────────
    a(9, 23.97, 9, 23.15)
    a(9, 22.45, 9, 22.15)
    a(9, 21.45, 9, 21.07)
    # diamond yes/no
    a(9, 20.25, 9, 20.15)                           # yes down
    ax.text(9.15, 20.2, "Yes", fontsize=8, color=C["green"])
    a(11.0, 20.8, 13.1, 20.8, "No", C["red"])       # no right
    a(9, 19.45, 9, 18.95)
    a(9, 18.25, 9, 18.05)
    a(9, 17.35, 9, 17.15)
    a(9, 16.45, 9, 16.05)
    a(9, 15.35, 9, 15.15)
    a(9, 14.45, 9, 14.25)
    a(9, 13.55, 9, 13.15)
    a(9, 12.45, 9, 12.25)
    a(9, 11.55, 9, 11.35)
    a(9, 10.65, 9, 10.25)
    a(9, 9.55,  9, 9.35)
    a(9, 8.65,  9, 8.45)
    a(9, 7.75,  9, 7.35)
    a(9, 6.75,  9, 6.45)
    a(9, 5.75,  9, 5.85)
    # ROUGE diamond yes/no
    a(9, 4.55, 9, 4.45)                             # yes down
    ax.text(9.15, 4.5, "Yes", fontsize=8, color=C["green"])
    a(11.5, 5.2, 13.1, 5.2, "No ↺", C["red"])      # no right
    a(9, 3.75, 9, 3.35)
    a(9, 2.65, 9, 2.35)
    a(9, 1.65, 9, 1.35)
    a(9, 0.67, 9, 0.50)

    # Legend
    lx, ly = 15.5, 9.5
    ax.text(lx + 0.5, ly, "Legend", fontsize=9, fontweight="bold",
            color=C["white"], ha="center")
    litems = [
        (C["green"],  "Start / End"),
        (C["teal"],   "Data Pipeline"),
        (C["sky"],    "Split"),
        (C["blue"],   "Modeling"),
        (C["violet"], "Base Model"),
        (C["amber"],  "Publish"),
        (C["orange"], "API / Serve"),
        (C["red"],    "Error path"),
    ]
    for i, (col, lbl) in enumerate(litems):
        r = FancyBboxPatch((lx, ly - 0.6 - i*0.5), 0.28, 0.3,
                           boxstyle="round,pad=0.02",
                           facecolor=col, edgecolor=C["white"],
                           linewidth=0.8, zorder=4)
        ax.add_patch(r)
        ax.text(lx + 0.42, ly - 0.6 - i*0.5 + 0.15, lbl,
                fontsize=8.5, color=C["light_slate"], va="center")

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT / "flow_diagram.pdf", dpi=200,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("  [OK] flow_diagram.pdf")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating diagrams...")
    make_class_diagram()
    make_architectural_diagram()
    make_flow_diagram()
    print("Done. All PDFs saved to:", OUT)
