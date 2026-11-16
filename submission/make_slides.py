"""Fill the organisers' submission template (Submission_Final_forGithub.pptx)
with Palmcast's content. Everything added is a native PowerPoint text box,
shape or table, so the deck stays fully editable. Only the template's own
colours are used: background 0A0E1A, panel 0D1B3E, cyan 00B4D8, mint 00E5B0,
amber F59E0B, text E8EDF5.

    python submission/make_slides.py --template Submission_Final_forGithub.pptx --out submission/Palmcast_XR_Hackathon_Track02.pptx
"""
import argparse
import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "submission" / "media"
REPO = "github.com/yugesh1111/xr"

BG, PANEL, CYAN, MINT, AMBER = "0A0E1A", "0D1B3E", "00B4D8", "00E5B0", "F59E0B"
TEXT, MUTED, PANEL2 = "E8EDF5", "8D9AB5", "13254F"
FONT = "Calibri"


def rgb(h):
    return RGBColor.from_string(h)


def box(slide, x, y, w, h, fill=PANEL, line=None, radius=True, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shp.adjustments[0] = 0.08
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if name:
        shp.name = name
    shp.text_frame.text = ""
    return shp


def text(slide, x, y, w, h, runs, size=13, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font=FONT, name=None, space_after=4, bullets=False):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if bullets:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.18)))
            pPr.set("indent", str(-Inches(0.16)))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            clr = pPr.makeelement(qn("a:buClr"), {})
            srgb = clr.makeelement(qn("a:srgbClr"), {"val": MUTED})
            clr.append(srgb)
            pPr.append(clr)
            pPr.append(bu)
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def picture(slide, path, x, y, w=None, h=None, name=None, line=PANEL2):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    if line:
        pic.line.color.rgb = rgb(line)
        pic.line.width = Pt(1)
    if name:
        pic.name = name
    return pic


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def set_text(shape, value, keep_first_run=True):
    """Replace a template text box's text, keeping its first run's formatting."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    r0 = p0.runs[0] if p0.runs else p0.add_run()
    r0.text = value
    for r in p0.runs[1:]:
        r.text = ""
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)


def card(slide, x, y, w, h, title, body, accent, title_size=14, body_size=12, name=None):
    box(slide, x, y, w, h, name=name)
    text(slide, x + 0.18, y + 0.14, w - 0.36, 0.4, [[(title, {"bold": True, "color": accent, "size": title_size})]])
    text(slide, x + 0.18, y + 0.52, w - 0.36, h - 0.62, body, size=body_size, color=TEXT)


def results_rows():
    rows = list(csv.DictReader(open(ROOT / "results" / "results.csv")))
    label = {"top1_accuracy": "Top-1 accuracy, unseen signers", "latency_p95": "Latency p95, per frame",
             "model_size": "Model size (INT8)", "accuracy_drop_robustness": "Robustness drop",
             "avg_power": "Average power", "thermal_drift_10min": "Thermal drift, 10 min"}
    out = []
    for r in rows:
        v = r["measured"]
        meas = f"{v} {r['unit']}" if v else "phone run pending"
        status = r["pass"] or "—"
        out.append((label.get(r["metric"], r["metric"]), r["target"].replace(">=", "≥ ").replace("<=", "≤ ") + " " + r["unit"],
                    meas, status, r["device"]))
    return out


def table(slide, x, y, w, col_w, rows, header, size=11, row_h=0.36):
    shp = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(rows) + 1)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    for r in range(len(rows) + 1):
        tbl.rows[r].height = Inches(row_h)
        vals = header if r == 0 else rows[r - 1]
        for c, v in enumerate(vals):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(PANEL if r == 0 else (BG if r % 2 else PANEL2))
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = ""
            run = tf.paragraphs[0].add_run()
            run.text = str(v)
            f = run.font
            f.name = FONT
            f.size = Pt(size)
            f.bold = r == 0 or (c == 3 and v == "PASS")
            col = MUTED if r == 0 else TEXT
            if r > 0 and c == 3:
                col = MINT if v == "PASS" else AMBER
            f.color.rgb = rgb(col)
            # thin grid lines in the panel colour
            tcPr = cell._tc.get_or_add_tcPr()
            for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = tcPr.makeelement(qn(edge), {"w": "6350"})
                sf = ln.makeelement(qn("a:solidFill"), {})
                sf.append(sf.makeelement(qn("a:srgbClr"), {"val": PANEL2}))
                ln.append(sf)
                tcPr.append(ln)
    return shp


def dup_slide(prs, src_idx, after_idx, header):
    """Copy a template section slide (header bar + title only, no pictures) and place it after `after_idx`."""
    import copy
    src = prs.slides[src_idx]
    new = prs.slides.add_slide(src.slide_layout)
    for shp in list(new.shapes):
        shp._element.getparent().remove(shp._element)
    for el in src.shapes._spTree.iterchildren():
        if el.tag.endswith("}sp") or el.tag.endswith("}grpSp") or el.tag.endswith("}cxnSp"):
            new.shapes._spTree.append(copy.deepcopy(el))
    bg = src._element.cSld.bg
    if bg is not None:
        new._element.cSld.insert(0, copy.deepcopy(bg))
    ids = prs.slides._sldIdLst
    el = ids[-1]
    ids.remove(el)
    ids.insert(after_idx + 1, el)
    for sh in new.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            set_text(sh, header)
            break
    return new


def stat(slide, x, y, w, h, number, label, color, num_size=36, name=None):
    box(slide, x, y, w, h, name=name)
    text(slide, x + 0.15, y + 0.12, w - 0.3, 0.75, [[(number, {"bold": True, "size": num_size, "color": color})]])
    text(slide, x + 0.15, y + 0.12 + num_size / 72 * 1.25, w - 0.3, h - 0.3 - num_size / 72 * 1.25, [[(label, {"size": 12})]])


def bar_chart(slide, x, y, w, h, cats, series, title, colors, vmin=None, vmax=None, fmt='0.0"%"', name=None, legend=False):
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
    cd = CategoryChartData()
    cd.categories = cats
    for sname, vals in series:
        cd.add_series(sname, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    if name:
        gf.name = name
    ch = gf.chart
    ch.font.name = FONT
    ch.font.size = Pt(11)
    ch.font.color.rgb = rgb(TEXT)
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    tr = ch.chart_title.text_frame.paragraphs[0].runs[0].font
    tr.size, tr.bold, tr.color.rgb, tr.name = Pt(13), True, rgb(TEXT), FONT
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.color.rgb = rgb(TEXT)
    va = ch.value_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(PANEL2)
    va.format.line.fill.background()
    va.tick_labels.font.color.rgb = rgb(MUTED)
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.number_format = fmt.replace("0.0", "0")
    va.tick_labels.number_format_is_linked = False
    ca = ch.category_axis
    ca.format.line.color.rgb = rgb(PANEL2)
    ca.tick_labels.font.color.rgb = rgb(TEXT)
    ca.tick_labels.font.size = Pt(10)
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10 if len(series) > 1 else 0
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(10)
    dl.font.color.rgb = rgb(TEXT)
    for i, s_ in enumerate(plot.series):
        s_.format.fill.solid()
        s_.format.fill.fore_color.rgb = rgb(colors[i % len(colors)])
        if len(series) == 1 and len(colors) == len(cats):
            for j, pt_col in enumerate(colors):
                pt = s_.points[j]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = rgb(pt_col)
    return gf


def build(template, out, team, college, members):
    prs = Presentation(template)
    # new section slides, made from the template's own headers:
    # after Architecture (index 3) -> MODEL & DATA; after Track Final Output (index 5 once inserted) -> RESULTS
    dup_slide(prs, 3, 3, "TRAINING & EVALUATION")
    dup_slide(prs, 3, 4, "PER-FRAME FLOW & EDGE CASES")
    dup_slide(prs, 6, 6, "TRACK REQUIREMENTS")
    dup_slide(prs, 6, 7, "RESULTS")
    dup_slide(prs, 6, 8, "IMPLEMENTATION")
    s = prs.slides
    # 0 title, 1 problem, 2 solution, 3 architecture, 4 training, 5 flow+edge cases, 6 track output,
    # 7 requirements, 8 results, 9 implementation, 10 differentiation, 11 demo, 12 brownie, 13 thank you

    # 1 - title
    sl = s[0]
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith("Track No"):
            set_text(sh, "Track 02")
        elif t.startswith("Team Name"):
            ps = sh.text_frame.paragraphs
            ps[0].runs[0].text = f"Team: {team}"
            for r in ps[0].runs[1:]:
                r.text = ""
            if len(ps) > 1 and ps[1].runs:
                ps[1].runs[0].text = college
                for r in ps[1].runs[1:]:
                    r.text = ""
        elif t.startswith("Members"):
            set_text(sh, "Members")
            sh.text_frame.paragraphs[0].runs[0].font.bold = True
            sh.text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(CYAN)
    text(sl, 5.62, 2.85, 3.6, 1.2, [[(m, {"size": 14})] for m in members], color=TEXT, name="Members list")
    text(sl, 0.55, 3.75, 6.6, 1.3,
         [[("Palmcast", {"bold": True, "size": 26, "color": MINT})],
          [("Sign language to text and speech, live, on the phone. No internet, no cloud.", {"size": 14})]],
         name="Product line")

    # 2 - the problem: three constraints, each with its number from the brief
    sl = s[1]
    text(sl, 0.45, 1.05, 9.1, 0.7,
         [[("Sign language conversations break down when only one side signs, and cloud translation sends video of "
            "people to a server.", {"size": 16})]], name="Problem statement")
    text(sl, 0.45, 1.95, 9.1, 0.4, [[("So the translator has to live on the phone, and that brings three hard limits:",
                                       {"size": 14, "color": MUTED})]], name="Lead-in")
    table(sl, 0.45, 2.45, 9.1, [2.2, 6.9], [
        ("Keep up", "A caption late by more than about 50 ms per frame makes the conversation feel out of sync."),
        ("Stay private", "The camera films a bystander. Their video must never leave the device: no cloud, no upload."),
        ("Stay small and cool", "Under 20 MB INT8 and about 500 mW, without heating up over a 10-minute conversation."),
    ], ["Limit", "Why it is hard"], size=13, row_h=0.5)
    text(sl, 0.45, 4.6, 9.1, 0.6,
         [[("On top of that the view is exocentric: the signer moves, stands at an angle, is badly lit, or walks partly "
            "out of frame.", {"size": 12, "color": MUTED})]], name="Conditions")

    # 3 - solution
    sl = s[2]
    picture(sl, MEDIA / "live_sentence_1.png", 0.45, 1.05, h=4.3, name="Screenshot live")
    picture(sl, MEDIA / "live_sentence_2.png", 2.5, 1.05, h=4.3, name="Screenshot sentence")
    text(sl, 4.65, 1.05, 4.95, 0.5, [[("Point the phone, read the words", {"bold": True, "size": 20, "color": MINT})]])
    text(sl, 4.65, 1.65, 4.95, 3.7, [
        [("Live captions ", {"bold": True}), ("of 100 ASL words (WLASL100), only when the model is sure", {})],
        [("Running sentence: ", {"bold": True}), ("signs join up; a pause ends it", {})],
        [("Speaks aloud ", {"bold": True}), ("with an offline voice", {})],
        [("Two-way: ", {"bold": True}), ("the hearing person taps Reply and talks; their words appear as captions", {})],
        [("Silent when idle: ", {"bold": True}), ("a 0.23 MB sign gate filters fidgets", {})],
        [("Offline: ", {"bold": True}), ("no internet permission at all", {})],
    ], size=14, bullets=True, space_after=8, name="Solution points")

    # 4 - architecture: the draw.io diagram, as large as the slide allows
    DIA = ROOT / "submission" / "diagrams"

    def diagram(sl, path, note):
        from PIL import Image
        iw, ih = Image.open(path).size
        h = 4.52
        w = h * iw / ih
        if w > 9.5:
            w, h = 9.5, 9.5 * ih / iw
        x = (10 - w) / 2
        box(sl, x - 0.06, 0.98, w + 0.12, h + 0.08, fill="FFFFFF", name="Diagram card")
        picture(sl, path, x, 1.02, w=w, h=h, line=None, name="Diagram")
        if note:
            sl.notes_slide.notes_text_frame.text = note + ". The PNG next to it is full resolution for zooming."

    sl = s[3]
    diagram(sl, DIA / "architecture_device.png", "Editable source: submission/diagrams/architecture_device.drawio (draw.io)")

    # 5 - training & evaluation diagram
    sl = s[4]
    diagram(sl, DIA / "architecture_training.png", "Editable source: submission/diagrams/architecture_training.drawio")

    # 6 - per-frame flow + edge cases
    sl = s[5]
    from PIL import Image
    iw, ih = Image.open(DIA / "flowchart.png").size
    fh = 4.45
    fw = fh * iw / ih
    box(sl, 0.39, 0.98, fw + 0.12, fh + 0.08, fill="FFFFFF", name="Flow card")
    picture(sl, DIA / "flowchart.png", 0.45, 1.02, w=fw, h=fh, line=None, name="Flowchart")
    edge = [("No hands in view", "Listening; every 2nd frame skipped"),
            ("Hands resting / fidget", "Movement-range check + sign gate → nothing"),
            ("Unsure sign", "Grey 'word?' + 2 alternatives, not spoken"),
            ("Out-of-vocabulary sign", "Sign gate / low confidence → nothing"),
            ("Too dark", "Torch or screen ring light; hint"),
            ("Blurry frame", "Adaptive sharpening before tracking"),
            ("Too close (face > 40%)", "\"Move the phone back\" hint"),
            ("Shoulders out of frame", "Virtual shoulders from the face"),
            ("Signer far away", "Trained on far copies: 77% at ⅓ size"),
            ("Left-handed signer", "Mirrored to the dominant hand"),
            ("Hand lost for a moment", "Gaps ≤ 6 frames interpolated"),
            ("Object passes in front", "Tested: −4.0 points"),
            ("Missing camera / voice / speech pack", "Falls back, tells the user")]
    tx = 0.45 + fw + 0.35
    table(sl, tx, 1.0, 9.55 - tx, [1.85, 9.55 - tx - 1.85], edge, ["Edge case", "What the app does"], size=9, row_h=0.31)

    # 6 - track final output
    sl = s[6]
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith("Track Final Output"):
            set_text(sh, "TRACK FINAL OUTPUT")
    rows = results_rows()
    table(sl, 0.45, 1.1, 5.9, [2.25, 1.05, 1.65, 0.95], [r[:4] for r in rows],
          ["Metric (results.csv)", "Target", "Measured", "Status"], size=11, row_h=0.4)
    text(sl, 0.45, 4.0, 5.9, 1.3, [
        [("Accuracy: 3 groups of signers removed from training (622 clips). Official WLASL100 test: 81.6%.", {"size": 10, "color": MUTED})],
        [("Latency and thermal: 10-minute sustained run on a laptop CPU (Apple M5), not a phone. Power needs the phone run.",
          {"size": 10, "color": MUTED})],
    ], space_after=3, name="Table notes")
    box(sl, 6.6, 1.1, 2.95, 3.25, name="Deliverables")
    text(sl, 6.78, 1.2, 2.6, 0.35, [[("Deliverables", {"bold": True, "size": 14, "color": AMBER})]])
    text(sl, 6.78, 1.6, 2.65, 2.7, [
        [("predict() ", {"bold": True}), ("in main.py", {})],
        [("predictions.jsonl ", {"bold": True}), ("201 test clips", {})],
        [("results.csv", {"bold": True})],
        [("INT8 ONNX ", {"bold": True}), ("+ conversion script", {})],
        [("README, Docker, ", {"bold": True}), ("Python 3.11", {})],
        [("Android APK", {"bold": True})],
        [("Demo video ", {"bold": True}), ("2:47 walkthrough", {})],
    ], size=12, bullets=True, space_after=5)

    # 7b - requirements from the track slides, with evidence
    sl = s[7]
    reqs = [("Fully on-device, exocentric: camera → text or audio", "Android app, back camera; captions + offline voice", "APK, demo video"),
            ("Moving heads, angled signers, low light", "Body-relative features; view augmentation; torch / ring light", "Robustness 4.64 pt"),
            ("On-device privacy, zero cloud streaming", "No INTERNET permission; nothing stored or sent", "AndroidManifest.xml"),
            ("Pose-invariant 3D hand features", "Hand shape in the hand's frame, location in the body's frame", "features.py, Features.kt"),
            ("Quantized TFLite / ONNX, < 20 MB INT8", "ONNX INT8 6.23 MB + gate 0.23 MB", "models/, export_onnx.py"),
            ("Latency ≤ 50 ms", "14.0 ms per frame p95; word model 5.4 ms", "sustained_cpu.json"),
            ("Captions or audio, two-way conversation", "Captions, voice, Reply turns speech into captions", "demo video"),
            ("predict(), predictions.jsonl, results.csv", "main.py; 201 test clips; 5 of 6 PASS", "results/"),
            ("Occlusions & HUD clarity (25%)", "Occlusion −4.0 pt; confidence bar, hints, conversation", "robustness json"),
            ("Python 3.11, requirements.txt, README, zip layout", "Checked on 3.11.15; make_submission.py", "submission/README.md")]
    table(sl, 0.4, 1.02, 9.2, [3.2, 3.9, 2.1], reqs, ["Track 02 asks for", "Palmcast", "Evidence"], size=10, row_h=0.38)

    # 8 - results charts
    sl = s[8]
    bar_chart(sl, 0.4, 1.05, 4.7, 2.3, ["Group A", "Group B", "Group C", "Official test"], [("Top-1", [80.8, 78.5, 74.9, 81.6])],
              "Accuracy on signers never seen (target 70%)", [MINT, MINT, MINT, CYAN], vmin=0, vmax=100, name="Chart holdout")
    bar_chart(sl, 5.0, 1.05, 4.6, 2.3, ["Clean", "Dark", "Blur", "Tilt 15°", "Low res", "Occluded"],
              [("Top-1", [80.1, 75.1, 72.3, 81.0, 72.8, 76.1])], "Bad video: mean drop 4.6 points (target ≤ 5)",
              [MINT, AMBER, AMBER, AMBER, AMBER, AMBER], vmin=0, vmax=100, name="Chart robustness")
    bar_chart(sl, 0.4, 3.4, 4.7, 2.0, ["Normal", "Arm's length", "Very close", "Far, ½ size", "Far, ⅓ size"],
              [("Top-1", [78.1, 71.9, 40.7, 76.9, 77.2])], "Near and far from the camera",
              [MINT, CYAN, AMBER, CYAN, CYAN], vmin=0, vmax=100, name="Chart near far")
    bar_chart(sl, 5.0, 3.4, 4.6, 2.0, ["Per frame", "Word model", "Limit"], [("ms", [14.0, 5.4, 50])],
              "Latency p95, ms (laptop CPU, 10-min run)", [MINT, MINT, MUTED], vmin=0, vmax=60, fmt='0.0', name="Chart latency")

    # 9 - implementation photos (real recordings)
    sl = s[9]
    # four steps, left to right, each with a numbered caption; result cards cropped so they can be read
    cap = lambda n, t: [[(f"{n}  ", {"bold": True, "color": MINT}), (t, {})]]
    picture(sl, MEDIA / "impl_live_help.png", 0.4, 1.02, h=3.45, name="Shot live")
    text(sl, 0.4, 4.52, 1.6, 0.6, cap(1, "Live camera: \"help\" read as help, 54%"), size=10, name="Caption 1")
    picture(sl, MEDIA / "impl_online_clip.png", 2.1, 1.02, h=3.45, name="Shot upload")
    text(sl, 2.1, 4.52, 1.6, 0.6, cap(2, "A web clip uploaded into the app, read frame by frame"), size=10, name="Caption 2")
    picture(sl, MEDIA / "impl_online_result.png", 3.8, 1.02, w=2.35, name="Shot result")
    text(sl, 3.8, 2.13, 2.35, 0.3, cap(3, "Its answer: basketball, 81%"), size=10, name="Caption 3")
    picture(sl, MEDIA / "impl_builtin_result.png", 3.8, 2.5, w=1.95, name="Shot builtin")
    text(sl, 3.8, 4.7, 2.4, 0.45, cap(4, "Built-in test: 9 of 10, same as Python"), size=10, name="Caption 4")
    picture(sl, MEDIA / "impl_terminal.png", 6.35, 1.02, w=3.25, name="Terminal")
    text(sl, 6.35, 3.4, 3.25, 0.9, cap(5, "predict() in Python on the same 9 web clips: 6 right. results.csv: 5 of 6 targets met "
                                          "(power needs a phone run)."), size=10, name="Caption 5")
    text(sl, 0.4, 5.12, 9.2, 0.3, [[("All screenshots come from screen recordings of the release APK on the Android emulator "
                                     "and a terminal session.", {"size": 9, "color": MUTED})]])

    # 10 - differentiation: numbers first
    sl = s[10]
    table(sl, 0.45, 1.1, 9.1, [3.4, 5.7], [
        ("Tested on clips of people already seen in training", "Whole signers held out: 78.1% on people the model never saw"),
        ("Any hand movement turns into a word", "Movement check, sign gate and jitter filter: no words from resting hands in 600 runs"),
        ("One word per button press", "Continuous: signs are cut at pauses and build a sentence (20 of 30 in the sentence test)"),
        ("Cloud API or a large model", "6.2 MB INT8 word model + 0.23 MB gate; the app has no internet permission"),
        ("The demo app and the evaluated code differ", "Same pipeline: the phone gives the Python answer on 10 of 10 clips"),
        ("Captions go one way only", "Reply: the hearing person speaks, the signer reads it as captions"),
    ], ["Common in sign-to-text demos", "What Palmcast does instead"], size=12, row_h=0.6)

    # 11 - demo & local setup
    sl = s[11]
    # the walkthrough video itself, playable in PowerPoint (click to play)
    vid = sl.shapes.add_movie(str(ROOT / "submission" / "Palmcast_walkthrough.mp4"), Inches(0.45), Inches(1.05),
                              Inches(5.25), Inches(5.25 * 9 / 16),
                              poster_frame_image=str(MEDIA / "walkthrough_poster.png"), mime_type="video/mp4")
    vid.name = "Walkthrough video"
    text(sl, 0.45, 4.1, 5.25, 1.3, [
        "Live phone camera · sentences forming · a web video uploaded · sentence test 20/30 · built-in test 9/10 · predict().",
        "Real screen recordings only, 2:47. File: submission/Palmcast_walkthrough.mp4.",
    ], size=11, bullets=True, space_after=4, name="Demo points")
    box(sl, 5.9, 1.05, 3.65, 4.25, name="Setup box")
    text(sl, 6.05, 1.12, 3.4, 0.35, [[("Local setup", {"bold": True, "size": 14, "color": AMBER})]])
    code = ["git clone https://" + REPO, "cd xr", "pip install -r", "  requirements-inference.txt",
            "python main.py --videos clips/", "  --out results/predictions.jsonl", "", "# or, no network:",
            "docker build -t palmcast .", "docker run --network none \\", "  -v $PWD/clips:/clips palmcast \\",
            "  --videos /clips", "", "# live pipeline, laptop webcam:", "pip install -r requirements-live.txt",
            "python live.py          # or --check"]
    text(sl, 6.05, 1.5, 3.45, 3.8, [[(c, {"font": "Courier New", "size": 9.5, "color": TEXT if not c.startswith("#") else MUTED})]
                                    for c in code], space_after=0, name="Setup commands")

    # 12 - brownie: gate chart + features
    sl = s[12]
    bar_chart(sl, 0.4, 1.05, 4.9, 4.3, ["Hands resting", "Fidgeting", "Other signs", "Real signs"],
              [("Without gate", [45.0, 33.1, 10.4, 65.6]), ("With gate", [7.2, 7.4, 8.8, 58.8])],
              "How often a word appears (%)", [MUTED, MINT], vmin=0, vmax=80, name="Chart gate", legend=True)
    text(sl, 5.55, 1.1, 4.0, 0.4, [[("Extras beyond the brief", {"bold": True, "size": 15, "color": AMBER})]], name="Extras title")
    text(sl, 5.55, 1.6, 4.0, 3.8, [
        [("Reply captions: ", {"bold": True}), ("the other person's speech becomes captions, on the phone", {})],
        [("Running sentence: ", {"bold": True}), ("words join up, a 4 s pause adds the full stop", {})],
        [("Offline voice: ", {"bold": True}), ("sure words are spoken with an installed voice", {})],
        [("Low light: ", {"bold": True}), ("torch, or a white ring on the screen", {})],
        [("Jitter filter: ", {"bold": True}), ("false words from resting hands under webcam jitter, 140 of 600 down to 0", {})],
        [("Measure mode: ", {"bold": True}), ("a 10-minute run logs latency, power and heat", {})],
    ], size=12, bullets=True, space_after=7, name="Extras")

    # 13 - thank you
    sl = s[13]
    for sh in sl.shapes:
        if sh.name == "Text 25":
            p = sh.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = f"Team WeCode · VIT Chennai · Palmcast · {REPO}"
            r.font.name = FONT
            r.font.size = Pt(12)
            r.font.color.rgb = rgb("FFFFFF")
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "Palmcast_XR_Hackathon_Track02.pptx"))
    ap.add_argument("--team", default="WeCode")
    ap.add_argument("--college", default="VIT Chennai")
    ap.add_argument("--members", nargs="+", default=["Amritha S (23BEC1368)", "Yugeshwaran P (23BEC1404)"])
    a = ap.parse_args()
    build(a.template, a.out, a.team, a.college, a.members)
