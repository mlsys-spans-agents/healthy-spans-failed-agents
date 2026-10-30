"""Find the red (TODO/TBD/CHECK) text runs in the paper PDF, grouped into contiguous blocks."""
import pymupdf as fitz


def is_red(c):
    r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
    return r > 150 and g < 60 and b < 60


def red_groups(doc):
    """Return a list of (page_no, [fragment rects], text, size) per contiguous red run."""
    out = []
    for pno, page in enumerate(doc):
        cur = None
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if is_red(s["color"]):
                        if cur is None:
                            cur = dict(page=pno, spans=[], size=s["size"])
                        cur["spans"].append(s)
                    elif s["text"].strip():
                        if cur:
                            out.append(cur)
                        cur = None
        if cur:
            out.append(cur)
    for g in out:  # merge spans into per-line fragments
        frags = []
        for s in g["spans"]:
            r = fitz.Rect(s["bbox"])
            if frags and abs(frags[-1].y0 - r.y0) < 2 and r.x0 - frags[-1].x1 < 3:
                frags[-1] |= r
            else:
                frags.append(r)
        g["frags"] = frags
        g["text"] = "".join(s["text"] for s in g["spans"])
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for i, g in enumerate(red_groups(fitz.open("healthy-spans-revised.pdf"))):
        print(i, "p", g["page"] + 1, len(g["frags"]), "frags", [round(f.width) for f in g["frags"]], repr(g["text"]))
