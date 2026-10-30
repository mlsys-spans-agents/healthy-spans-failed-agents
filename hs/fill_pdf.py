"""Write healthy-spans-filled.pdf: every red/orange TODO, TBD and CHECK in the paper is removed and the
final text is set in black in its place. Each edited paragraph is re-set from the edit point to its end
(justified), so no gaps are left. Numbers come from results/results.json."""
import json
import os
import re
import sys

import pymupdf as fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
FONT = fitz.Font(fontfile="C:/Windows/Fonts/times.ttf")  # Times with Greek glyphs
BASE14 = fitz.Font("tiro")  # built-in Times: clean text extraction for ASCII words
BODY, TABLE = 9.96, 8.97


def font_for(w):
    return BASE14 if w.isascii() else FONT


def write_word(tw, x, base, w, size):
    """Set a word in runs: built-in Times for ASCII, the TTF only for other characters (e.g. tau)."""
    import itertools
    for asc, run in itertools.groupby(w, key=str.isascii):
        run = "".join(run)
        f = BASE14 if asc else FONT
        tw.append((x, base), run, font=f, fontsize=size)
        x += f.text_length(run, size)
COLS = [(63.0, 297.0), (315.0, 549.0)]

tau, trail, by, dg = R["tau"], R["trail"], R["trail_by_benchmark"], R["diagnosis"]
so_t, so_r = tau["SIZE-ONLY"], trail["SIZE-ONLY"]
clf_t, clf_r = tau["Metadata classifier"], trail["Metadata classifier"]
jt = R["judge_truncation"]


# words that end a line with a real (compound) hyphen rather than a TeX break hyphen
COMPOUND = {"tail-", "error-", "first-", "size-", "high-", "end-", "multi-", "long-", "two-", "per-",
            "trace-", "agent-", "outcome-", "span-", "low-", "real-", "decision-", "well-", "equal-", "last-"}


def marked(c):
    return c in (0xBF0000, 0xCC6600)


def col_of(x):
    return 0 if x < 306 else 1


# --------------------------------------------------------------------------- page model
def body_lines(page):
    """Body-text lines (size ~9.96) in reading order: dicts with col, base, spans, x0, x1, y0, y1."""
    acc = {}
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if not s["text"]:
                    continue
                c = col_of(s["bbox"][0])
                base = s["origin"][1]
                k = next((k for k in acc if k[0] == c and abs(k[1] - base) < 1.5), (c, base))
                acc.setdefault(k, []).append(s)
    lines = []
    for (c, base), spans in acc.items():
        spans.sort(key=lambda s: s["bbox"][0])
        if max(s["size"] for s in spans) < 9.5 or base < 60 or base > 740:  # captions, tables, header, page no.
            continue
        r = fitz.Rect(spans[0]["bbox"])
        for s in spans:
            r |= fitz.Rect(s["bbox"])
        lines.append(dict(col=c, base=base, spans=spans, x0=r.x0, x1=r.x1, y0=r.y0, y1=r.y1))
    lines.sort(key=lambda d: (d["col"], d["base"]))
    return lines


def paragraph_from(lines, i):
    """Indices of the lines from line i to the end of its paragraph (may cross into the next column)."""
    out, j = [i], i
    while j + 1 < len(lines):
        cur, nxt = lines[j], lines[j + 1]
        if cur["x1"] < COLS[cur["col"]][1] - 4:  # short line ends the paragraph
            break
        if nxt["col"] == cur["col"] and nxt["base"] - cur["base"] > 14:
            break
        if nxt["x0"] > COLS[nxt["col"]][0] + 4:  # indented line starts a new paragraph
            break
        j += 1
        out.append(j)
    return out


def line_text(line, start_x=None):
    out, prev = "", None
    for s in line["spans"]:
        if start_x is not None and s["bbox"][0] < start_x - 0.5:
            continue
        if prev is not None and s["bbox"][0] - prev["bbox"][2] > 1.5                 and not out.endswith(" ") and not s["text"].startswith(" "):
            out += " "  # TeX positions words apart without a space glyph
        out += s["text"].replace("­", "-")
        prev = s
    return out


def join_lines(parts):
    text = ""
    for p in parts:
        p = p.strip()
        a, b = text[-2:-1], p[:1]
        stem = text.rsplit(" ", 1)[-1].lower()
        if text.endswith("-") and a.isalpha() and b.isalpha() and a.islower() == b.islower()                 and stem not in COMPOUND:
            text = text[:-1] + p  # undo end-of-line hyphenation
        elif text.endswith("-") or not text:
            text += p
        else:
            text += " " + p
    return text


# --------------------------------------------------------------------------- re-setting
def reset(page, rects, text, size=BODY):
    """Erase rects and set text across them, justified except the last line; shrink slightly if needed."""
    for r in rects:
        page.add_redact_annot(fitz.Rect(r.x0 + (0.6 if r._mid else -0.3), r.y0 + 1.2, r.x1 + 0.5, r.y1 - 1.0), fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    words = text.split()
    sz = size
    while True:
        placed, k = [], 0
        for r, base in rects_b(rects):
            line = []
            x0 = r.x0 + (FONT.text_length(" ", sz) if r._lead else 0.0)
            while k < len(words) and x0 + FONT.text_length(" ".join(line + [words[k]]), sz) <= r.x1 + 0.3:
                line.append(words[k])
                k += 1
            placed.append((r, base, line))
        if k == len(words) or sz < 8.5:
            break
        sz -= 0.1
    tw = fitz.TextWriter(page.rect)
    last = max((n for n, (_, _, l) in enumerate(placed) if l), default=-1)
    for n, (r, base, line) in enumerate(placed):
        if not line:
            continue
        natural = FONT.text_length(" ".join(line), sz)
        gap = (r.x1 - r.x0 - (FONT.text_length(" ", sz) if r._lead else 0.0) - natural) / (len(line) - 1) if (n < last and len(line) > 1) else 0.0
        if gap > 6.0:  # too few words to justify: set ragged
            gap = 0.0
        x = r.x0 + (FONT.text_length(" ", sz) if r._lead else 0.0)
        for w in line:
            write_word(tw, x, base, w, sz)
            x += FONT.text_length(w + " ", sz) + gap
    tw.write_text(page, color=(0, 0, 0))
    return k == len(words)


def rects_b(rects):
    return [(r, r._base) for r in rects]


def edit(page, anchor, subs):
    """Re-set the paragraph containing the anchor (first marked span, or the first span containing the
    anchor word) from the anchor to the paragraph end, after applying regex substitutions `subs`."""
    lines = body_lines(page)
    hit, lead = None, False
    for i, l in enumerate(lines):
        for s in l["spans"]:
            if anchor == "@marked":
                if marked(s["color"]):
                    hit = (i, s["bbox"][0])
                    lead = s["text"].startswith(" ")
            if hit:
                break
        if anchor != "@marked" and anchor in line_text(l):
            rs = [r for r in page.search_for(anchor) if abs(r.y0 - l["y0"]) < 3 and col_of(r.x0) == l["col"]]
            hit = (i, rs[0].x0)
        if hit:
            break
    i, x = hit
    idx = paragraph_from(lines, i)
    text = join_lines([line_text(lines[idx[0]], x)] + [line_text(lines[j]) for j in idx[1:]])
    new = text
    for pat, rep in subs:
        new, n = re.subn(pat, rep, new, count=1, flags=re.S)
        if n != 1:
            raise ValueError(f"pattern not found: {pat!r}\nin: {text}")
    if new[:1] in ",.;:)":
        lead = False
    rects = []
    for n, j in enumerate(idx):
        l = lines[j]
        r = fitz.Rect(x if n == 0 else l["x0"], l["y0"], COLS[l["col"]][1], l["y1"])
        r._base = l["base"]
        r._mid = n == 0 and x > COLS[l["col"]][0] + 1
        r._lead = n == 0 and lead
        rects.append(r)
    if not reset(page, rects, new):
        raise ValueError("replacement does not fit: " + new)


def put(page, x_right, base, s, size=TABLE):
    tw = fitz.TextWriter(page.rect)
    tw.append((x_right - FONT.text_length(s, size), base), s, font=font_for(s), fontsize=size)
    tw.write_text(page, color=(0, 0, 0))


def clear_cells(page):
    """Erase marked table cells; return their baselines."""
    cells = [(fitz.Rect(s["bbox"]), s["origin"][1]) for b in page.get_text("dict")["blocks"]
             for l in b.get("lines", []) for s in l["spans"] if marked(s["color"]) and "TBD" in s["text"]]
    for r, _ in cells:
        page.add_redact_annot(fitz.Rect(r.x0 - 0.3, r.y0 + 0.5, r.x1 + 0.3, r.y1 - 0.5), fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return sorted({round(b, 1) for _, b in cells})


# --------------------------------------------------------------------------- the edits
def main():
    doc = fitz.open(os.path.join(ROOT, "healthy-spans-revised.pdf"))
    T = r"\s*\[(?:TODO|CHECK):?"  # opening of a marker

    # p3 [CHECK]
    edit(doc[2], "@marked", [(T + r".*?code\]", " TRAIL is reweighted the same way from its own failure share (74.3%).")])

    # p4 marker TODO
    edit(doc[3], "@marked", [(T + r" either.*?Section 9\.”\]",
                              " The marker also shows which step numbers are missing; we discuss its effect in Section 9.")])

    # p5 Table 2 SIZE-ONLY row
    p = doc[4]
    base, = clear_cells(p)
    for xr, s in [(468.5, f"{so_t['recall']}%"), (492.3, f"{so_t['ere']:.2f}"),
                  (524.6, f"{so_r['recall']}%"), (549.0, f"{so_r['ere']:.2f}")]:
        put(p, xr, base, s)

    # p6: benchmark TODO, then the size paragraph (X, Y, X, Y, keep-one)
    p = doc[5]
    # correction: SIZE-ONLY ignores the outcome and keeps 79.1%, above the classifier's low end of 63.5%
    edit(p, "Even the low end", [(r"well above every rule that ignores the outcome",
                                  "well above the error, latency, cost and anomaly rules (though not the size-only "
                                  "policy below)"),
                                 (T + r" report AUC.*?apart\.”\]",
                         f" Within GAIA the classifier's AUC is {by['pooled_oof_auc_GAIA']:.2f} and within SWE-bench "
                         f"{by['pooled_oof_auc_SWE']:.2f} (only {by['n_SWE'][0]} traces). Benchmark identity alone has "
                         f"an AUC of {1 - by['auc_benchmark_identity']:.2f}, and with folds grouped by benchmark the "
                         f"classifier's AUC falls to {by['benchmark_grouped_auc']:.2f}, so part of its gain comes from "
                         "telling the benchmarks apart.")])
    edit(p, "@marked", [
        (T + r"\s*X\]", f" {so_t['recall']}"),
        (T + r"\s*Y\]", f" {so_r['recall']}"),
        (T + r"\s*X\]", f" {clf_t['recall_equal_count']}"),
        (T + r"\s*Y\]", f" {clf_r['recall_equal_count']}"),
        (T + r" keep one:.*?remains\.”\]",
         f" On TRAIL the size-only policy comes close to the classifier ({so_r['recall']}% against 82.7%), so much "
         "of the TRAIL gain reflects which runs are small rather than which runs failed; on τ-bench the gap remains "
         f"({so_t['recall']}% against 38.9%)."),
    ])

    # p7 judge truncation
    edit(doc[6], "@marked", [(T + r" optional.*?run\.”\]",
                              f" Between {jt['name + input/output']['share_at_limit']:.0f}% and "
                              f"{jt['span JSON']['share_at_limit']:.0f}% of TRAIL runs reached the 40,000-character "
                              "limit, so the judge often saw only part of the run.")])

    # p8: Table 6 cells, agent sentence, chance X/Y
    p = doc[7]
    chance = [dg["ERROR+8"]["chance_view_on_kept"], dg["Last 8"]["chance_view_on_kept"],
              dg["First 30%"]["chance_view_on_kept"]]
    for n, base in enumerate(clear_cells(p)):
        put(p, 241.0, base, "–")
        put(p, 266.0, base, "–")
        if n % 2 == 1:
            put(p, 299.3, base, f"{chance[n // 2]}%")
    edit(p, "It shows", [(r"It shows" + T + r" the same / a smaller\] drop:.*?from the view\.",
                          "It falls too: over all 176 failures the investigator names the responsible agent 64.2% "
                          "of the time from the full trace and 40.9% from the error-anchored view (Table 5)."),
                         (T + r"\s*X\]", f" {dg['ERROR+8']['chance_view_on_kept']}"),
                         (T + r"\s*Y\]", f" {dg['ERROR+8']['chance_full_on_kept']}")])

    # p9 collector
    p = doc[8]
    edit(p, "@marked", [(T + r" rerun.*?settings”\]",
                         " v0.120.1, the release current when we built the harness, with the processor's default "
                         "settings,")])
    edit(p, "exceeded", [(r"exceeded" + T + r" threshold\], chosen" + T + r"\s*how\]\)",
                          "exceeded a fixed threshold)")])
    edit(p, "@marked", [(T + r"\s*confirm the window was scaled in this run\]", "")])

    # p11
    p = doc[10]
    edit(p, "agent accuracy,", [(r"agent accuracy, a chance baseline, and the effect of removing the marker\." + T + r".*?not run\]",
                           "agent accuracy and a chance baseline.")])
    # artifact link: overwrite only the marked fragments so the monospace words keep their font
    allsp = [sp for bl in p.get_text("dict")["blocks"] for l in bl.get("lines", []) for sp in l["spans"]]
    spans = [sp for sp in allsp if marked(sp["color"])]
    line2 = [sp for sp in spans if sp["origin"][1] > spans[0]["origin"][1] + 2]
    close = next(sp for sp in allsp if sp["text"].startswith(")") and abs(sp["origin"][1] - line2[0]["origin"][1]) < 1
                 and sp["bbox"][0] >= line2[-1]["bbox"][2] - 1)
    for sp in spans + [close]:
        r = fitz.Rect(sp["bbox"])
        x1 = r.x0 + FONT.text_length(")", BODY) + 0.3 if sp is close else r.x1 + 0.3
        p.add_redact_annot(fitz.Rect(r.x0 + 0.3, r.y0 + 1.2, x1, r.y1 - 1.0), fill=(1, 1, 1))
    p.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    x0, x1 = line2[0]["bbox"][0], close["bbox"][0] + FONT.text_length(")", BODY)
    words = "provided as supplementary material)".split()
    gap = (x1 - x0 - FONT.text_length(" ".join(words), BODY)) / (len(words) - 1)
    tw = fitz.TextWriter(p.rect)
    x = x0
    for w in words:
        tw.append((x, line2[0]["origin"][1]), w, font=font_for(w), fontsize=BODY)
        x += FONT.text_length(w + " ", BODY) + gap
    tw.write_text(p, color=(0, 0, 0))

    left = [(n + 1, s["text"]) for n, pg in enumerate(doc) for b in pg.get_text("dict")["blocks"]
            for l in b.get("lines", []) for s in l["spans"] if marked(s["color"])]
    for n, pg in enumerate(doc):
        for w in ("TODO", "TBD", "[CHECK"):
            if pg.search_for(w):
                left.append((n + 1, w))
    out = os.path.join(ROOT, "healthy-spans-filled.pdf")
    doc.save(out, garbage=3, deflate=True)
    print("wrote", out, "| marked text left:", left)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
