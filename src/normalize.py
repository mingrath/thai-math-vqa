r"""
Answer normalization + extraction + voting for Thai Math VQA.

The competition metric is EXACT-MATCH after a normalization pipeline applied to
BOTH prediction and gold (see TASK.md):

    - lowercase + strip whitespace
    - Thai digits ๐–๙ -> 0–9
    - strip `$` and units (องศา, หน่วย, บาท, …)
    - expand LaTeX: \frac, \sqrt, \pi
    - strip `{ } \ ,` + canonical integer (2.0 -> 2)

`normalize()` below replicates that pipeline (fused with battle-tested rules from
Qwen2.5-Math `strip_string` and the Minerva normalizer) so we can score locally
and so voting collapses equivalent surface forms (e.g. "2.0" == "2").

IMPORTANT: for the actual submission we still emit a representative *raw* answer
(not the normalized string) so the grader's own normalizer can do its job; we only
use `normalize()` to decide which cluster of answers won the vote.
"""
import re
from collections import Counter

# --- Thai digit map (U+0E50..U+0E59) ---------------------------------------
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# --- Thai units to strip (LONGEST-FIRST so ตารางเซนติเมตร strips before เซนติเมตร) ---
_THAI_UNITS_RAW = [
    # area (square ...) — must precede the base length units
    "ตารางกิโลเมตร", "ตารางเซนติเมตร", "ตารางมิลลิเมตร", "ตารางเมตร", "ตารางหน่วย",
    "ลูกบาศก์เซนติเมตร", "ลูกบาศก์เมตร", "ตารางนิ้ว", "ตารางฟุต",
    # length
    "กิโลเมตร", "เซนติเมตร", "มิลลิเมตร", "เมตร", "นิ้ว", "ฟุต", "หลา", "ไมล์",
    # mass / weight
    "กิโลกรัม", "มิลลิกรัม", "กรัม", "ตัน", "ขีด",
    # volume
    "ลูกบาศก์", "ลิตร", "มิลลิลิตร",
    # angle / temperature
    "องศาเซลเซียส", "องศาฟาเรนไฮต์", "องศา", "เรเดียน",
    # money / counting / misc
    "บาท", "สตางค์", "หน่วย", "เปอร์เซ็นต์", "ร้อยละ",
    "คน", "ชิ้น", "อัน", "ใบ", "ลูก", "ตัว", "แท่ง", "เส้น",
    # time
    "วินาที", "นาที", "ชั่วโมง", "วัน", "ปี", "เดือน", "สัปดาห์",
]
THAI_UNITS = sorted(set(_THAI_UNITS_RAW), key=len, reverse=True)

# LaTeX spacing/text cruft to drop
_LATEX_CRUFT = [
    r"\\text\{[^}]*\}", r"\\mathrm\{[^}]*\}", r"\\mathbf\{[^}]*\}",
    r"\\;", r"\\,", r"\\!", r"\\ ", r"\\left", r"\\right",
    r"\^\{?\\circ\}?", r"\\degree",
]


def _fix_sqrt(s: str) -> str:
    # \sqrt3 -> \sqrt{3}
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def _expand_simple_frac(s: str) -> str:
    # \frac12 -> \frac{1}{2}  (shorthand without braces)
    return re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)


def normalize(answer) -> str:
    """Canonicalize an answer string to the competition's exact-match form."""
    if answer is None:
        return ""
    s = str(answer)

    # --- Thai pre-pass ---
    s = s.translate(THAI_DIGITS)                       # ๒ -> 2
    s = re.sub(r"(ข้อ|คำตอบ|ตอบ|ได้|เท่ากับ|คือ)", "", s)  # answer-label noise
    for u in THAI_UNITS:                               # strip Thai units (longest first)
        s = s.replace(u, "")

    # --- competition pipeline ---
    s = s.lower().strip()
    s = s.replace("\n", "").rstrip(".")

    # strip $ / % symbols
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")

    # frac family + shorthand expansion (BEFORE we strip braces)
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = _expand_simple_frac(s)
    s = _fix_sqrt(s)

    # drop LaTeX spacing/text/degree cruft
    for pat in _LATEX_CRUFT:
        s = re.sub(pat, "", s)

    # keep \pi as the literal token "pi" (do NOT substitute 3.14)
    s = s.replace("\\pi", "pi")

    # strip braces / backslashes / commas / spaces (LAST, per stated pipeline)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")

    # leading-dot fix: .5 -> 0.5
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0" + s[1:]

    # canonical numbers: 2.0 -> 2 , 3.50 -> 3.5 , "2." -> "2"
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    s = re.sub(r"(?<=\d)\.$", "", s)

    return s


# --- final-answer extraction ----------------------------------------------
def _extract_boxed(text: str):
    """Return the content of the LAST \\boxed{...}, brace-balanced (handles nesting)."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth, buf = 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    return "".join(buf).strip()


_ANSWER_LABEL = re.compile(
    r"(?:final answer|answer|คำตอบ|ตอบ)\s*[:：=]?\s*(.+)", re.IGNORECASE
)
_LAST_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str:
    """Pull the final short answer from a model's raw (possibly CoT) output."""
    if not text:
        return ""
    text = text.strip()

    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    # "Final answer: X" / "คำตอบ: X" — take the last such line
    labels = _ANSWER_LABEL.findall(text)
    if labels:
        cand = labels[-1].strip()
        # trim trailing prose/period
        cand = cand.split("\n")[0].strip().rstrip(".")
        return cand

    # fallback: last line; if it ends in "... = <expr/number>", take the RHS
    last_line = text.split("\n")[-1].strip()
    if "=" in last_line:
        rhs = last_line.rsplit("=", 1)[-1].strip().rstrip(".")
        if rhs:
            return rhs
    if last_line:
        return last_line
    nums = _LAST_NUMBER.findall(text)
    return nums[-1] if nums else text


# --- self-consistency voting ----------------------------------------------
def majority_vote(raw_answers):
    """
    Given a list of raw answer strings (already extracted from model outputs),
    vote on their NORMALIZED form and return the most common *raw* representative
    of the winning cluster. Returns (winning_raw, vote_count, total).
    """
    cleaned = [a for a in raw_answers if a is not None and str(a).strip() != ""]
    if not cleaned:
        return "", 0, 0

    norm_counts = Counter(normalize(a) for a in cleaned)
    best_norm, count = norm_counts.most_common(1)[0]

    # among raws that normalize to best_norm, pick the most frequent raw surface form
    raws_in_cluster = [a for a in cleaned if normalize(a) == best_norm]
    best_raw = Counter(raws_in_cluster).most_common(1)[0][0]
    return best_raw, count, len(cleaned)


def equiv(pred, gold) -> bool:
    """Official-style exact match after normalization."""
    return normalize(pred) == normalize(gold)


# --- self-test -------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("20 ตารางเซนติเมตร", "20"),
        ("๓๖", "36"),
        ("2.0", "2"),
        ("2.50", "2.5"),
        ("$\\frac{17}{10}$", "frac{17}{10}".replace("{", "").replace("}", "")),  # frac1710
        ("$6\\sqrt{3}$", "6sqrt3"),
        ("30 องศา", "30"),
        (".5", "0.5"),
        ("ตอบ 76", "76"),
        ("\\boxed{45}", "45"),  # note: normalize doesn't extract; extract_answer does
    ]
    print("== normalize ==")
    for raw, expect in cases:
        got = normalize(raw)
        flag = "OK " if got == expect else "XX "
        print(f"  {flag} {raw!r:30} -> {got!r:15} (expect {expect!r})")

    print("== extract_answer ==")
    samples = [
        "Let me solve... therefore the area is\nFinal answer: \\boxed{20 ตารางเซนติเมตร}",
        "คิดเป็น ... คำตอบ: 76",
        "...so x = 45",
    ]
    for s in samples:
        print(f"  {extract_answer(s)!r}  <=  {s!r}")

    print("== majority_vote ==")
    votes = ["2.0", "2", "สอง", "2", "3"]
    print("  ", majority_vote(votes))
