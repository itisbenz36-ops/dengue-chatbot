# -*- coding: utf-8 -*-
"""ฐานข้อมูล SQLite รองรับผู้ใช้หลายคนพร้อมกัน"""
import os
import json
import random
import sqlite3
import datetime as dt
from contextlib import contextmanager

DB_PATH = os.environ.get("DENGUE_DB", "dengue_research.db")


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=30000")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS participants (
            pid         TEXT PRIMARY KEY,
            created_at  TEXT,
            group_code  TEXT,
            sex         TEXT,
            age_group   TEXT,
            education   TEXT,
            role        TEXT,
            experience  TEXT,
            village     TEXT,
            profiled    INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS tests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pid         TEXT,
            phase       TEXT,
            k_score     INTEGER,
            b_total     INTEGER,
            b_mean      REAL,
            answers     TEXT,
            created_at  TEXT,
            UNIQUE(pid, phase)
        );
        CREATE TABLE IF NOT EXISTS chats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pid         TEXT,
            turn        INTEGER,
            question    TEXT,
            topic_id    TEXT,
            topic       TEXT,
            source      TEXT,
            score       REAL,
            created_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS satisfaction (
            pid         TEXT PRIMARY KEY,
            scores      TEXT,
            mean        REAL,
            comment     TEXT,
            created_at  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chats_pid ON chats(pid);
        """)


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def new_pid() -> str:
    """สร้างรหัสสั้นอ่านง่าย ไม่มีตัวอักษรที่สับสน"""
    chars = "ACDEFGHJKLMNPQRTUVWXY3456789"
    with _conn() as c:
        for _ in range(50):
            pid = "DF" + "".join(random.choice(chars) for _ in range(5))
            if not c.execute("SELECT 1 FROM participants WHERE pid=?", (pid,)).fetchone():
                c.execute("INSERT INTO participants(pid, created_at) VALUES(?,?)",
                          (pid, _now()))
                return pid
    raise RuntimeError("ไม่สามารถสร้างรหัสได้")


def participant_exists(pid: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM participants WHERE pid=?", (pid,)).fetchone() is not None


def save_profile(pid: str, d: dict):
    with _conn() as c:
        c.execute("""UPDATE participants SET group_code=?, sex=?, age_group=?,
                     education=?, role=?, experience=?, village=?, profiled=1
                     WHERE pid=?""",
                  (d.get("group_code"), d["sex"], d["age_group"], d["education"],
                   d["role"], d["experience"], d.get("village"), pid))


def save_test(pid, phase, k_score, b_total, b_mean, answers: dict):
    with _conn() as c:
        c.execute("""INSERT OR REPLACE INTO tests
                     (pid, phase, k_score, b_total, b_mean, answers, created_at)
                     VALUES(?,?,?,?,?,?,?)""",
                  (pid, phase, k_score, b_total, b_mean,
                   json.dumps(answers, ensure_ascii=False), _now()))


def save_chat(pid, turn, question, topic_id, topic, source, score):
    with _conn() as c:
        c.execute("""INSERT INTO chats
                     (pid, turn, question, topic_id, topic, source, score, created_at)
                     VALUES(?,?,?,?,?,?,?,?)""",
                  (pid, turn, question, topic_id, topic, source, score, _now()))


def save_satisfaction(pid, scores: dict, mean, comment):
    with _conn() as c:
        c.execute("""INSERT OR REPLACE INTO satisfaction
                     (pid, scores, mean, comment, created_at) VALUES(?,?,?,?,?)""",
                  (pid, json.dumps(scores, ensure_ascii=False), mean, comment, _now()))


def get_chat_history(pid: str):
    with _conn() as c:
        rows = c.execute("""SELECT question, topic FROM chats
                            WHERE pid=? ORDER BY id""", (pid,)).fetchall()
    return [dict(r) for r in rows]


def chat_count(pid: str) -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM chats WHERE pid=?", (pid,)).fetchone()[0]


def get_stage(pid: str) -> str:
    """คำนวณขั้นตอนปัจจุบันจากฐานข้อมูล ทำให้รีเฟรชแล้วไม่หาย"""
    with _conn() as c:
        p = c.execute("SELECT profiled FROM participants WHERE pid=?", (pid,)).fetchone()
        if p is None:
            return "consent"
        if not p["profiled"]:
            return "profile"
        phases = {r["phase"] for r in
                  c.execute("SELECT phase FROM tests WHERE pid=?", (pid,)).fetchall()}
        if "pre" not in phases:
            return "pre"
        n = c.execute("SELECT COUNT(*) FROM chats WHERE pid=?", (pid,)).fetchone()[0]
        if "post" not in phases:
            return "chat" if n < 5 else "chat_done"
        if not c.execute("SELECT 1 FROM satisfaction WHERE pid=?", (pid,)).fetchone():
            return "sat"
        return "done"


def get_result(pid: str) -> dict:
    with _conn() as c:
        out = {}
        for r in c.execute("SELECT * FROM tests WHERE pid=?", (pid,)).fetchall():
            out[r["phase"]] = dict(r)
        s = c.execute("SELECT * FROM satisfaction WHERE pid=?", (pid,)).fetchone()
        out["sat"] = dict(s) if s else None
        out["n_chat"] = c.execute("SELECT COUNT(*) FROM chats WHERE pid=?",
                                  (pid,)).fetchone()[0]
    return out


def export_wide():
    """รวมทุกตารางเป็นแถวเดียวต่อผู้เข้าร่วม พร้อมนำเข้า SPSS"""
    import pandas as pd
    with _conn() as c:
        parts = pd.read_sql_query("SELECT * FROM participants", c)
        tests = pd.read_sql_query("SELECT * FROM tests", c)
        sat = pd.read_sql_query("SELECT * FROM satisfaction", c)
        chats = pd.read_sql_query("SELECT * FROM chats", c)

    if parts.empty:
        return parts, chats

    df = parts.drop(columns=["profiled"])
    for phase in ("pre", "post"):
        t = tests[tests.phase == phase].copy()
        if t.empty:
            continue
        exp = t["answers"].apply(json.loads).apply(pd.Series)
        exp.columns = [f"{phase}_{c_}" for c_ in exp.columns]
        t = pd.concat([t[["pid", "k_score", "b_total", "b_mean"]].reset_index(drop=True),
                       exp.reset_index(drop=True)], axis=1)
        t = t.rename(columns={"k_score": f"{phase}_k_score",
                              "b_total": f"{phase}_b_total",
                              "b_mean": f"{phase}_b_mean"})
        df = df.merge(t, on="pid", how="left")

    if not sat.empty:
        exp = sat["scores"].apply(json.loads).apply(pd.Series)
        s = pd.concat([sat[["pid", "mean", "comment"]].reset_index(drop=True),
                       exp.reset_index(drop=True)], axis=1)
        s = s.rename(columns={"mean": "sat_mean"})
        df = df.merge(s, on="pid", how="left")

    if not chats.empty:
        cnt = chats.groupby("pid").size().rename("n_questions").reset_index()
        df = df.merge(cnt, on="pid", how="left")

    if "pre_k_score" in df and "post_k_score" in df:
        df["k_gain"] = df["post_k_score"] - df["pre_k_score"]
    if "pre_b_mean" in df and "post_b_mean" in df:
        df["b_gain"] = (df["post_b_mean"] - df["pre_b_mean"]).round(2)

    return df, chats