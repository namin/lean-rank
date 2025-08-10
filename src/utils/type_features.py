import re, hashlib, numpy as np

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_\.']*")
ARROW = re.compile(r"->|→")
FORALL = re.compile(r"∀|Pi")
IMPLIES = re.compile(r"→|->")
LBRACK = re.compile(r"\[")
RBRACK = re.compile(r"\]")

def hash_bucket(s: str, buckets: int = 128) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % buckets

def head_symbol(type_text: str) -> str:
    ids = IDENT.findall(type_text)
    return ids[-1] if ids else ""

def featurize_type(type_text: str, buckets: int = 128):
    if type_text is None: type_text = ""
    length = len(type_text)
    n_arrow = len(ARROW.findall(type_text))
    n_forall = len(FORALL.findall(type_text))
    n_impl = len(IMPLIES.findall(type_text))
    n_lbrack = len(LBRACK.findall(type_text))
    n_rbrack = len(RBRACK.findall(type_text))

    tri = np.zeros(buckets, dtype=np.float32)
    s = type_text
    for i in range(len(s)-2):
        tri[hash_bucket(s[i:i+3], buckets)] += 1.0

    head = head_symbol(type_text)
    head_b = np.zeros(buckets, dtype=np.float32)
    if head:
        head_b[hash_bucket(head, buckets)] = 1.0

    base = np.array([length, n_arrow, n_forall, n_impl, n_lbrack, n_rbrack], dtype=np.float32)
    return base, tri, head_b

def feature_names(buckets: int = 128):
    names = ["len", "n_arrow", "n_forall", "n_impl", "n_lbrack", "n_rbrack"]
    names += [f"tri_{i}" for i in range(buckets)]
    names += [f"head_{i}" for i in range(buckets)]
    return names
