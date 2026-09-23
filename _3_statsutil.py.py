# -*- coding: utf-8 -*-
"""สถิติพื้นฐานเขียนเอง ไม่ต้องพึ่ง scipy"""
import math


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_pvalue_two_tailed(t: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_ttest(pre: list, post: list) -> dict:
    """คืนค่า t, df, p, mean diff, Cohen's dz"""
    pairs = [(a, b) for a, b in zip(pre, post)
             if a is not None and b is not None
             and not (isinstance(a, float) and math.isnan(a))
             and not (isinstance(b, float) and math.isnan(b))]
    n = len(pairs)
    if n < 2:
        return {"n": n, "error": "ข้อมูลไม่พอ ต้องมีอย่างน้อย 2 คู่"}

    d = [b - a for a, b in pairs]
    md = sum(d) / n
    var = sum((x - md) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return {"n": n, "mean_diff": md, "sd_diff": 0.0,
                "t": float("inf") if md else 0.0, "df": n - 1,
                "p": 0.0 if md else 1.0, "d": float("inf") if md else 0.0}

    se = sd / math.sqrt(n)
    t = md / se
    return {
        "n": n, "mean_pre": sum(a for a, _ in pairs) / n,
        "mean_post": sum(b for _, b in pairs) / n,
        "mean_diff": md, "sd_diff": sd, "t": t, "df": n - 1,
        "p": t_pvalue_two_tailed(t, n - 1), "d": md / sd,
    }


def effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "น้อยมาก"
    if a < 0.5:
        return "น้อย"
    if a < 0.8:
        return "ปานกลาง"
    return "มาก"


def cronbach_alpha(matrix: list) -> float:
    """matrix = list ของ list คะแนนรายข้อ (แถว = คน, คอลัมน์ = ข้อ)"""
    rows = [r for r in matrix if all(v is not None for v in r)]
    n = len(rows)
    if n < 2 or not rows[0]:
        return float("nan")
    k = len(rows[0])

    def var(vals):
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

    item_var = sum(var([r[j] for r in rows]) for j in range(k))
    total_var = var([sum(r) for r in rows])
    if total_var == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - item_var / total_var)