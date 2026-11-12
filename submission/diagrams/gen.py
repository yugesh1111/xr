"""Writes architecture.drawio and flowchart.drawio (open and edit them in draw.io).
Plain style: white background, grey boxes, black text, one accent (teal) for the model parts."""
from xml.sax.saxutils import escape

BOX = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#4A4A4A;fontColor=#1A1A1A;fontSize=13;fontFamily=Helvetica;arcSize=8;"
MODEL = "rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F4F1;strokeColor=#0F766E;fontColor=#0B3B36;fontSize=13;fontFamily=Helvetica;arcSize=8;fontStyle=1;"
GROUP = "rounded=1;whiteSpace=wrap;html=1;fillColor=#F7F7F7;strokeColor=#9A9A9A;dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;fontSize=14;fontStyle=1;fontColor=#333333;fontFamily=Helvetica;arcSize=3;"
DIAMOND = "rhombus;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#4A4A4A;fontColor=#1A1A1A;fontSize=12;fontFamily=Helvetica;"
TERM = "rounded=1;whiteSpace=wrap;html=1;arcSize=50;fillColor=#F2F2F2;strokeColor=#4A4A4A;fontColor=#1A1A1A;fontSize=13;fontFamily=Helvetica;"
OUT = "rounded=1;whiteSpace=wrap;html=1;fillColor=#0F766E;strokeColor=#0F766E;fontColor=#FFFFFF;fontSize=13;fontFamily=Helvetica;fontStyle=1;arcSize=8;"
NOTE = "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize=11;fontColor=#555555;fontFamily=Helvetica;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#4A4A4A;fontSize=11;fontColor=#333333;fontFamily=Helvetica;"
DEDGE = EDGE + "dashed=1;"


class D:
    def __init__(s):
        s.cells, s.n = [], 2

    def v(s, label, x, y, w, h, style=BOX, parent="1"):
        s.n += 1
        if "\n" in label and "rhombus" not in style:  # first line bold, rest normal
            head, rest = label.split("\n", 1)
            label = "<b>" + head + "</b><br>" + rest.replace("\n", "<br>")
        else:
            label = label.replace("\n", "<br>")
        i = f"c{s.n}"
        s.cells.append(f'<mxCell id="{i}" value="{escape(label, {'"': '&quot;'})}" style="{style}" vertex="1" parent="{parent}">'
                       f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def e(s, a, b, label="", style=EDGE, pts=None):
        s.n += 1
        geo = '<mxGeometry relative="1" as="geometry">'
        if pts:
            geo += "<Array as=\"points\">" + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts) + "</Array>"
        geo += "</mxGeometry>"
        s.cells.append(f'<mxCell id="c{s.n}" value="{escape(label, {'"': '&quot;'})}" style="{style}" edge="1" parent="1" source="{a}" target="{b}">{geo}</mxCell>')

    def save(s, path, name, w, h):
        body = "".join(s.cells)
        xml = (f'<mxfile host="drawio"><diagram name="{name}" id="{name}"><mxGraphModel dx="{w}" dy="{h}" grid="1" gridSize="10" '
               f'guides="1" page="1" pageWidth="{w}" pageHeight="{h}" background="#FFFFFF" math="0" shadow="0"><root>'
               f'<mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')
        open(path, "w").write(xml)


# ---------------- architecture
d = D()
d.v("Offline, on a computer (training + export)", 20, 20, 1560, 200, GROUP)
a1 = d.v("WLASL100\n1,466 videos, 100 words\n+ Kaggle ASL Signs 94,477 (pre-training)", 40, 70, 250, 110)
a2 = d.v("MediaPipe Holistic\nlandmark extraction\n(scripts/extract_landmarks.py)", 330, 70, 220, 110)
a3 = d.v("Features, 239 per frame\nhand shape, orientation, location,\nmotion, face (echogesture/features.py)", 590, 70, 260, 110)
a4 = d.v("Training: 4 networks\nbase, near/far, degraded, pre-trained\n+ sign gate (train.py, train_gate.py)", 890, 70, 270, 110)
a5 = d.v("INT8 export\nONNX static per-channel\n(export_onnx.py)", 1200, 70, 200, 110, MODEL)
a6 = d.v("models/\nmodel_int8.onnx 6.2 MB\ngate_int8.onnx 0.23 MB", 1430, 70, 130, 110, MODEL)
for x, y in [(a1, a2), (a2, a3), (a3, a4), (a4, a5), (a5, a6)]:
    d.e(x, y)

d.v("On the phone (Android app, fully offline, no internet permission)", 20, 260, 1560, 470, GROUP)
p1 = d.v("CameraX\n720p frames\nback camera (exocentric)", 40, 320, 190, 90)
p2 = d.v("Low-light check\ntorch / screen ring light", 40, 450, 190, 70)
p3 = d.v("MediaPipe Holistic\n88 body, hand, face points\n(GPU, CPU fallback)", 270, 320, 210, 90)
p4 = d.v("Adaptive sharpening\nonly when the frame is blurry", 270, 450, 210, 70)
p5 = d.v("Sign detector\nstart / end of a sign\n(0.5 s pause, movement range)", 520, 320, 220, 90)
p6 = d.v("Features (Kotlin)\nsame as Python, tested to ~1e-6", 780, 320, 220, 90)
p7 = d.v("Sign gate\nINT8, 0.12 ms\nsign or not?", 1040, 320, 170, 90, MODEL)
p8 = d.v("Word model\n4-net INT8 ensemble\nONNX Runtime", 1250, 320, 170, 90, MODEL)
p9 = d.v("Caption rule\n≥ 45% and 20 points ahead\nelse grey guess or nothing", 1250, 450, 300, 80)
q1 = d.v("Sentence builder\nwords → sentence, pause → full stop", 880, 450, 330, 80)
q2 = d.v("HUD: caption, confidence bar,\nconversation, hints", 520, 600, 260, 80, OUT)
q3 = d.v("Offline voice (TTS)", 820, 600, 200, 80, OUT)
q4 = d.v("Reply: microphone →\non-device speech recognition", 1060, 600, 250, 80)
q5 = d.v("Measurement mode\nlatency, power, heat → CSV", 1350, 600, 200, 80)
d.e(p1, p3); d.e(p1, p2); d.e(p3, p4, style=DEDGE); d.e(p3, p5); d.e(p5, p6); d.e(p6, p7); d.e(p7, p8)
d.e(p8, p9); d.e(p9, q1); d.e(q1, q2); d.e(q1, q3); d.e(q4, q2, pts=[(1185, 710), (650, 710)])
d.e(a6, p8, "assets", DEDGE, pts=[(1495, 240), (1335, 240)])

d.v("Evaluation (the brief's predict())", 20, 770, 1560, 140, GROUP)
e1 = d.v("main.py predict(video)\n→ {clip_id, prediction, top5,\nconfidence, latency_ms}", 40, 810, 300, 85)
e2 = d.v("results/predictions.jsonl\n201 official test clips", 380, 810, 250, 85)
e3 = d.v("results/results.csv\nsix metrics vs targets", 670, 810, 230, 85)
e4 = d.v("Docker, --network none\nPython 3.11", 940, 810, 220, 85)
d.e(e1, e2); d.e(e2, e3)
d.v("Uses the same models/ files as the app", 1200, 815, 340, 40, NOTE)
d.save("architecture.drawio", "Architecture", 1600, 930)

# ---------------- flowchart (what happens to every camera frame)
f = D()
s0 = f.v("Camera frame", 340, 20, 200, 50, TERM)
c1 = f.v("Too dark?", 365, 100, 150, 80, DIAMOND)
l1 = f.v("Light on: torch or\nscreen ring light", 620, 110, 200, 60)
k1 = f.v("Landmarks (MediaPipe Holistic)", 315, 210, 250, 55)
c2 = f.v("Hand in view?", 365, 295, 150, 80, DIAMOND)
i1 = f.v("Idle: skip every\n2nd frame, no model", 620, 305, 200, 60)
c3 = f.v("Sign ended?\n0.5 s still or hands\ndown, or 3.5 s", 350, 405, 180, 110, DIAMOND)
w1 = f.v("Keep collecting\nframes", 620, 430, 200, 60)
c4 = f.v("Moved enough?\nwrist ≥ 0.2 or finger\n≥ 0.25 face heights", 350, 545, 180, 110, DIAMOND)
n1 = f.v("Ignore\n(fidget, resting hand)", 620, 570, 200, 60)
x1 = f.v("Features → sign gate", 315, 685, 250, 55, MODEL)
c5 = f.v("Looks like a sign?", 355, 770, 170, 90, DIAMOND)
n2 = f.v("Show nothing", 620, 785, 200, 60)
m1 = f.v("Word model (INT8)", 315, 890, 250, 55, MODEL)
c6 = f.v("≥ 45% and 20\npoints ahead?", 355, 975, 170, 90, DIAMOND)
o1 = f.v("Show word, add to\nsentence, speak it", 40, 990, 230, 60, OUT)
c7 = f.v("≥ 25%?", 620, 980, 140, 80, DIAMOND)
o2 = f.v("Grey 'word?' + 2\nalternatives (not spoken)", 820, 990, 220, 60)
n3 = f.v("Show nothing", 820, 1090, 220, 50)
f.e(s0, c1); f.e(c1, l1, "yes"); f.e(c1, k1, "no"); f.e(l1, k1, pts=[(720, 237)])
f.e(k1, c2); f.e(c2, i1, "no"); f.e(c2, c3, "yes")
f.e(c3, w1, "no"); f.e(c3, c4, "yes"); f.e(c4, n1, "no"); f.e(c4, x1, "yes")
f.e(x1, c5); f.e(c5, n2, "no"); f.e(c5, m1, "yes"); f.e(m1, c6)
f.e(c6, o1, "yes"); f.e(c6, c7, "no"); f.e(c7, o2, "yes"); f.e(c7, n3, "no", pts=[(690, 1115)])
f.save("flowchart.drawio", "Per-frame flow", 1080, 1160)
print("ok")
