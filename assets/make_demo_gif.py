"""Render assets/demo.gif: an animated overview of the amusingkernels loop.

085 frames use real numbers from the 085_geglu_activation campaign
(run-20261102T173923Z-a1e9bd): versions are written, signature-checked, and
evaled; every eval reports time vs roofline SOL. The fastest full pass wins
and is frozen as solution.json; durable findings go to learnings.md.

Critic, stagnation, and prediction frames use real artifacts from the
005_gemm_n256_k7168 dual-model campaign (run-20261108T000801Z-a74036): the
Gemini diagnosis gist, the retrieved KernelWiki technique id, and the
predicted-vs-measured B200 latency pair.

    python assets/make_demo_gif.py   # needs pillow
"""

import os

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (250, 250, 247)
INK = (38, 38, 36)
MUTED = (120, 120, 114)
LINE = (205, 205, 198)
ACCENT = (37, 99, 235)
OK = (22, 142, 86)
BAD = (200, 60, 50)
GOLD = (196, 142, 20)
PURPLE = (124, 58, 237)
CARD = (255, 255, 255)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_H, F_B, F_S = font(26, True), font(16, True), font(15), font(12)

STAGES = [
    (80, "Problem", ["085_geglu_activation", "16 workloads"]),
    (222, "Agent", ["one tool per turn", "muse-spark-1.3"]),
    (360, "write", ["write_kernel", "v1, v2, …"]),
    (498, "build", ["build_solution", "DPS signature check"]),
    (648, "eval", ["sol-execbench, GB10", "correctness, then time"]),
    (830, "Best", ["fastest full pass", "frozen solution.json"]),
]
WRITE, BUILD, EVAL, BEST_COL = 2, 3, 4, 5
LANES = [205, 265, 325, 385]

# Real 085 campaign outcomes. FATE = last stage reached; EVAL = measured label.
VERSIONS = [
    ("v1", "triton", EVAL, "0.766ms · 78%", "ok"),
    ("v2", "triton", EVAL, "eval failed", "bad"),
    ("v3", "triton", EVAL, "0.711ms · 84%", "ok"),
    ("v4", "triton", EVAL, "0.738ms · 81%", "ok"),
]
BEST = 2


def base(run_label, caption, stages=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((32, 22), "amusingkernels", font=F_TITLE, fill=INK)
    title_right = d.textbbox((32, 22), "amusingkernels", font=F_TITLE)[2]
    d.text((title_right + 16, 31), "kernel-optimization agent for SOL-ExecBench",
           font=F_B, fill=MUTED)
    d.text((W - 32, 31), run_label, font=F_H, fill=ACCENT, anchor="ra")
    for x, title, sub in (STAGES if stages else []):
        d.text((x, 92), title, font=F_H, fill=INK, anchor="ma")
        for i, s in enumerate(sub):
            d.text((x, 115 + 17 * i), s, font=F_S, fill=MUTED, anchor="ma")
        d.line([(x, 170), (x, 420)], fill=LINE, width=1)
    d.rounded_rectangle([32, 452, W - 32, 508], radius=10, fill=CARD, outline=LINE)
    d.text((W // 2, 480), caption, font=F_B, fill=INK, anchor="mm")
    return img, d


def problem_card(d):
    d.rounded_rectangle([25, 265, 135, 325], radius=8, fill=CARD, outline=INK, width=2)
    d.text((80, 287), "085_geglu", font=F_S, fill=INK, anchor="mm")
    d.text((80, 305), "definition +", font=F_S, fill=MUTED, anchor="mm")
    d.text((80, 305 + 14), "workloads", font=F_S, fill=MUTED, anchor="mm")


def version_card(d, x, y, label, state):
    color = {"new": ACCENT, "ok": OK, "bad": BAD, "best": GOLD}[state]
    d.rounded_rectangle([x - 40, y - 19, x + 40, y + 19], radius=7, fill=CARD,
                        outline=color, width=2)
    mark = {"new": "", "ok": " ✓", "bad": " ✗", "best": " ★"}[state]
    d.text((x, y), f"{label}{mark}", font=F_S, fill=color, anchor="mm")


def eval_label(d, y, text, color):
    d.text((694, y), text, font=F_S, fill=color, anchor="lm")


def arrow(d, start, end, color, width=2):
    d.line([start, end], fill=color, width=width)
    (x0, y0), (x1, y1) = start, end
    length = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d.polygon([(x1, y1), (x1 - 11 * ux + 5 * uy, y1 - 11 * uy - 5 * ux),
               (x1 - 11 * ux - 5 * uy, y1 - 11 * uy + 5 * ux)], fill=color)


def card(d, box, title, lines, color, title_fill=None):
    d.rounded_rectangle(box, radius=10, fill=CARD, outline=color, width=2)
    cx = (box[0] + box[2]) // 2
    d.text((cx, box[1] + 22), title, font=F_H, fill=title_fill or color, anchor="mm")
    for i, line in enumerate(lines):
        d.text((cx, box[1] + 52 + 20 * i), line, font=F_S, fill=INK, anchor="mm")


def learnings_frame():
    img, d = base("across runs", "Durable findings become learnings.md, read by the next run",
                  stages=False)
    sources = [("v2 ✗  tanh intrinsic failed eval", BAD),
               ("v3 ✓  0.711ms, 84% of SOL", OK),
               ("v1 ✓  0.766ms baseline", OK),
               ("cuda ✗  toolchain not ready", BAD)]
    for i, (label, color) in enumerate(sources):
        y = 150 + 62 * i
        d.rounded_rectangle([40, y - 20, 268, y + 20], radius=7, fill=CARD,
                            outline=color, width=2)
        d.text((154, y), label, font=F_S, fill=color, anchor="mm")
        arrow(d, (270, y), (336, 205 + 26 * i), MUTED)
    d.rounded_rectangle([340, 188, 566, 298], radius=10, fill=CARD, outline=ACCENT, width=2)
    d.text((453, 210), "learnings.md", font=F_H, fill=ACCENT, anchor="mm")
    d.text((453, 237), "manual tanh over intrinsic", font=F_S, fill=INK, anchor="mm")
    d.text((453, 258), "Triton-first on this box", font=F_S, fill=INK, anchor="mm")
    d.text((453, 279), "large shapes carry the ms", font=F_S, fill=INK, anchor="mm")
    arrow(d, (568, 243), (604, 243), ACCENT, 3)
    d.rounded_rectangle([606, 112, 930, 388], radius=10, fill=CARD, outline=LINE)
    d.text((622, 128), "Next run objective", font=F_H, fill=INK)
    rules = ["Prior learnings from earlier runs:",
             "• manual tanh over intrinsic",
             "• Triton-first on this box",
             "• large shapes carry the ms",
             "",
             "Optimize '083_attention_…",
             "Write a kernel run(…) that is",
             "correct first, then fast."]
    for i, rule in enumerate(rules):
        fill = ACCENT if i == 0 else INK
        d.text((622, 164 + 26 * i), rule, font=F_S, fill=fill)
    d.text((453, 325), "one lesson per record_lesson call", font=F_S, fill=MUTED, anchor="mm")
    return img


def critic_frame():
    img, d = base("005_gemm_n256_k7168", "A failed eval gets a Gemini second opinion",
                  stages=False)
    card(d, [40, 170, 250, 300], "v1 ✗ 16/17",
         ["0.012ms on B200", "1 profiler flake"], BAD)
    arrow(d, (252, 235), (298, 235), MUTED, 3)
    card(d, [300, 150, 620, 320], "Gemini second opinion",
         ["RUNTIME_ERROR is a profiler flake,", "not a kernel bug", "",
          "diagnosis appended for", "the driver's next turn"], PURPLE)
    arrow(d, (622, 235), (668, 235), PURPLE, 3)
    card(d, [670, 170, 920, 300], "driver's fix",
         ["v2: Triton GEMM", "BM32 · BN128 · BK64"], ACCENT)
    d.text((480, 360), "the critic never calls tools and never stops the run",
           font=F_S, fill=MUTED, anchor="mm")
    return img


def stagnation_frame():
    img, d = base("005_gemm_n256_k7168", "Stalled with no new best: technique + fresh proposal",
                  stages=False)
    card(d, [40, 180, 240, 290], "stalled ×3",
         ["no new best", "for 3 evals"], GOLD)
    arrow(d, (242, 235), (288, 235), MUTED, 3)
    card(d, [290, 150, 590, 320], "[KERNELWIKI …stages]",
         ["technique-pipeline-stages", "compute-bound → num_stages ↑", "",
          "rotates past tried pages"], ACCENT)
    arrow(d, (592, 235), (638, 235), MUTED, 3)
    card(d, [640, 150, 920, 320], "fresh proposal",
         ["Gemini: one different", "approach, not a tweak", "",
          "guidance appended"], PURPLE)
    d.text((480, 360), "wiki checkout missing → block skipped silently",
           font=F_S, fill=MUTED, anchor="mm")
    return img


def prediction_frame():
    img, d = base("005_gemm_n256_k7168", "Triton GEMMs get a B200 prediction; pairs calibrate it",
                  stages=False)
    card(d, [40, 170, 240, 300], "v1 triton",
         ["tiles 128×256×64", "from launch site"], ACCENT)
    arrow(d, (242, 235), (288, 235), MUTED, 3)
    card(d, [290, 150, 570, 320], "B200 evaluator",
         ["Petri net → 123µs", "SMEM-constrained", "advisory only"], GOLD)
    arrow(d, (572, 235), (618, 235), MUTED, 3)
    card(d, [620, 150, 920, 320], "Modal B200",
         ["51µs · 17/17 PASSED", "pair → calibration.jsonl", "",
          "helper calibrate"], OK)
    d.text((480, 360), "prediction never blocks or fails an eval",
           font=F_S, fill=MUTED, anchor="mm")
    return img


def frames():
    out = []
    run = "085_geglu_activation"
    img, d = base(run, "Start from a SOL-ExecBench problem: definition + workloads")
    problem_card(d)
    out.append((img, 1000))

    img, d = base(run, "The agent writes kernel versions, one tool per turn")
    problem_card(d)
    for i, y in enumerate(LANES):
        d.line([(135, 295), (182, y)], fill=MUTED, width=2)
        version_card(d, STAGES[1][0], y, VERSIONS[i][0], "new")
    out.append((img, 1400))

    captions = {WRITE: "write_kernel saves each version (Triton-first)",
                BUILD: "build_solution validates the DPS signature, writes solution.json"}
    for stage in (WRITE, BUILD):
        img, d = base(run, captions[stage])
        problem_card(d)
        for i, y in enumerate(LANES):
            version_card(d, STAGES[stage][0], y, VERSIONS[i][0], "new")
        out.append((img, 1200))

    img, d = base(run, "run_eval: correctness first, then time vs GB10 roofline SOL")
    problem_card(d)
    for i, y in enumerate(LANES):
        _, _, _, label, state = VERSIONS[i]
        version_card(d, STAGES[EVAL][0], y, VERSIONS[i][0], state)
        eval_label(d, y, ("✗ " if state == "bad" else "") + label,
                   BAD if state == "bad" else OK)
    out.append((img, 2200))

    out.append((critic_frame(), 2200))

    img, d = base(run, "Fastest full pass wins — v3, frozen as solution.json")
    problem_card(d)
    for i, y in enumerate(LANES):
        _, _, _, label, state = VERSIONS[i]
        card = "best" if i == BEST else state
        version_card(d, STAGES[EVAL][0], y, VERSIONS[i][0], card)
        if i == BEST:
            d.text((STAGES[EVAL][0] - 48, y), "★ " + label, font=F_S, fill=GOLD, anchor="rm")
        else:
            eval_label(d, y, ("✗ " if state == "bad" else "") + label,
                       BAD if state == "bad" else OK)
    y, x = LANES[BEST], STAGES[EVAL][0]
    drop = x + 150
    d.line([(x + 40, y + 12), (drop, y + 12), (drop, 432), (830, 432), (830, 405)],
           fill=GOLD, width=3)
    d.polygon([(830, 405), (823, 417), (837, 417)], fill=GOLD)
    d.rounded_rectangle([770, 340, 890, 372], radius=7, fill=CARD, outline=GOLD, width=2)
    d.text((830, 356), "solution.json", font=F_S, fill=GOLD, anchor="mm")
    out.append((img, 2000))

    out.append((stagnation_frame(), 2200))
    out.append((prediction_frame(), 2200))

    out.append((learnings_frame(), 2600))

    img, d = base("", "Correctness first, time vs SOL always, learning across runs",
                  stages=False)
    lines = [("Problems", "SOL-ExecBench L1 + FlashInfer-Bench GEMMs"),
             ("Loop", "one tool per turn, saved, pause/resume"),
             ("Models", "Muse drives · Gemini reviews + proposes"),
             ("Tools", "read → write → build → eval → trace → lesson"),
             ("Stall", "KernelWiki technique after 3 evals idle"),
             ("Predict", "B200 latency + calibration pairs"),
             ("Score", "time vs roofline SOL on every eval"),
             ("Output", "frozen solution.json, Modal B200 traces")]
    for i, (k, v) in enumerate(lines):
        d.text((90, 130 + 38 * i), k, font=F_H, fill=ACCENT)
        d.text((260, 131 + 38 * i), v, font=F_B, fill=INK)
    out.append((img, 3000))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fr = frames()
    imgs = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f, _ in fr]
    path = os.path.join(here, "demo.gif")
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=[ms for _, ms in fr], loop=0, optimize=True)
    print(f"wrote {path} ({len(imgs)} frames, {os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
