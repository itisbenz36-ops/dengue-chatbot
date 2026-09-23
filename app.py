# -*- coding: utf-8 -*-
"""
แชตบอต AI ให้ความรู้และส่งเสริมพฤติกรรมการป้องกันโรคไข้เลือดออก
พร้อมระบบเก็บข้อมูลวิจัย  |  รัน: streamlit run app.py
"""
import os
import json
import datetime as dt

import pandas as pd
import streamlit as st

import db
import kb
from stats_util import paired_ttest, effect_label, cronbach_alpha

# ══════════════════════ ค่าคงที่ของงานวิจัย ══════════════════════
MIN_QUESTIONS = 5          # จำนวนคำถามขั้นต่ำก่อนทำ post-test
APP_TITLE = "น้องลายเดียว"

KNOWLEDGE_ITEMS = [
    ("K01", "โรคไข้เลือดออกมียุงลายเป็นพาหะนำโรค", True),
    ("K02", "ยุงลายกัดคนในเวลากลางคืนเป็นหลัก", False),
    ("K03", "ยุงลายวางไข่ในน้ำใสที่ขังนิ่ง เช่น แจกัน จานรองกระถาง", True),
    ("K04", "ผู้ที่เคยป่วยไข้เลือดออกแล้วจะไม่ป่วยซ้ำอีก", False),
    ("K05", "เมื่อสงสัยไข้เลือดออก ห้ามใช้ยาแอสไพรินและไอบูโพรเฟน", True),
    ("K06", "ช่วงที่ไข้เริ่มลดลงเป็นระยะที่ปลอดภัยแล้ว ไม่ต้องเฝ้าระวัง", False),
    ("K07", "มาตรการ 3 เก็บ คือ เก็บบ้าน เก็บขยะ และเก็บน้ำ", True),
    ("K08", "การพ่นหมอกควันเพียงอย่างเดียวกำจัดลูกน้ำยุงลายได้", False),
    ("K09", "ควรเปลี่ยนน้ำในแจกันและภาชนะขังน้ำทุก 7 วัน", True),
    ("K10", "ปวดท้องรุนแรงและอาเจียนไม่หยุด เป็นสัญญาณอันตรายที่ต้องรีบพบแพทย์", True),
    ("K11", "ไข้เลือดออกติดต่อจากคนสู่คนโดยการไอหรือจาม", False),
    ("K12", "ไข่ยุงลายที่ติดผนังภาชนะสามารถทนแห้งอยู่ได้นานหลายเดือน", True),
    ("K13", "การขัดล้างผนังภาชนะช่วยกำจัดไข่ยุงลายได้ดีกว่าการเทน้ำทิ้งอย่างเดียว", True),
    ("K14", "ไข้เลือดออกพบเฉพาะในเด็กเท่านั้น", False),
    ("K15", "ผู้ป่วยไข้เลือดออกควรนอนในมุ้งเพื่อป้องกันการแพร่เชื้อสู่ผู้อื่น", True),
]

BEHAVIOR_ITEMS = [
    ("B01", "สำรวจภาชนะที่มีน้ำขังรอบบ้านเพื่อหาลูกน้ำยุงลาย"),
    ("B02", "เปลี่ยนน้ำในแจกันหรือภาชนะขังน้ำอย่างน้อยสัปดาห์ละครั้ง"),
    ("B03", "ปิดฝาภาชนะเก็บน้ำให้มิดชิด"),
    ("B04", "เก็บกวาดเศษขยะและภาชนะที่อาจมีน้ำขังรอบบ้าน"),
    ("B05", "ใส่ทรายกำจัดลูกน้ำหรือเลี้ยงปลากินลูกน้ำในภาชนะเก็บน้ำ"),
    ("B06", "ใส่เกลือหรือทรายในจานรองกระถางต้นไม้เพื่อไม่ให้มีน้ำขัง"),
    ("B07", "ขัดล้างผนังภาชนะเก็บน้ำ ไม่ใช่เพียงเทน้ำทิ้ง"),
    ("B08", "จัดบ้านให้โปร่งโล่ง เก็บเสื้อผ้าไม่ให้เป็นที่เกาะพักของยุง"),
    ("B09", "ใช้ยาทากันยุงหรือสวมเสื้อแขนยาวป้องกันยุงกัดในเวลากลางวัน"),
    ("B10", "นอนกางมุ้งหรืออยู่ในห้องที่มีมุ้งลวด"),
    ("B11", "แนะนำคนในครอบครัวหรือเพื่อนบ้านให้กำจัดแหล่งเพาะพันธุ์ยุงลาย"),
    ("B12", "เข้าร่วมกิจกรรมกำจัดลูกน้ำยุงลายของชุมชน"),
]

BEHAVIOR_SCALE = {
    "ปฏิบัติเป็นประจำ (5-7 วัน/สัปดาห์)": 3,
    "ปฏิบัติบางครั้ง (1-4 วัน/สัปดาห์)": 2,
    "ไม่เคยปฏิบัติเลย": 1,
}

SATISFACTION_ITEMS = [
    ("S1", "เนื้อหาที่ได้รับชัดเจนและเข้าใจง่าย"),
    ("S2", "แชตบอตตอบตรงกับสิ่งที่ท่านต้องการทราบ"),
    ("S3", "รูปแบบการใช้งานสะดวก ไม่ซับซ้อน"),
    ("S4", "ความรู้ที่ได้สามารถนำไปปฏิบัติจริงที่บ้านได้"),
    ("S5", "ทำให้ท่านตระหนักถึงความสำคัญของการป้องกันโรคมากขึ้น"),
    ("S6", "ท่านจะแนะนำแชตบอตนี้ให้ผู้อื่นใช้งาน"),
]
LIKERT5 = {"มากที่สุด": 5, "มาก": 4, "ปานกลาง": 3, "น้อย": 2, "น้อยที่สุด": 1}


def interpret_k(score, full=len(KNOWLEDGE_ITEMS)):
    p = score / full * 100
    return "ระดับสูง" if p >= 80 else ("ระดับปานกลาง" if p >= 60 else "ระดับต่ำ")


def interpret_b(mean):
    return "ระดับดี" if mean >= 2.34 else ("ระดับปานกลาง" if mean >= 1.67 else "ควรปรับปรุง")


# ══════════════════════ ตั้งค่าหน้าเว็บ ══════════════════════
st.set_page_config(page_title=f"{APP_TITLE} | ป้องกันไข้เลือดออก",
                   page_icon="🦟", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 17px; }
  .stRadio label, .stSelectbox label { font-size: 16px !important; }
  div[data-testid="stChatMessage"] { font-size: 16.5px; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }
  .stButton button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

db.init_db()
S = st.session_state


# ══════════════════════ จัดการรหัสผู้เข้าร่วม ══════════════════════
def current_pid():
    if "pid" in S and S.pid:
        return S.pid
    qp = st.query_params.get("pid")
    if qp and db.participant_exists(qp):
        S.pid = qp
        return qp
    return None


def set_pid(pid):
    S.pid = pid
    st.query_params["pid"] = pid


def goto(stage):
    S.override_stage = stage
    st.rerun()


def resolve_stage(pid):
    if S.get("override_stage"):
        s = S.pop("override_stage")
        return s
    return db.get_stage(pid)


# ══════════════════════ หน้าเข้าสู่ระบบ ══════════════════════
def page_landing():
    st.title("🦟 น้องลายเดียว")
    st.caption("แชตบอต AI ให้ความรู้การป้องกันโรคไข้เลือดออกในชุมชน")

    tab1, tab2 = st.tabs(["🆕 เข้าร่วมครั้งแรก", "🔄 ทำต่อจากเดิม"])

    with tab1:
        st.markdown("""
### เอกสารชี้แจงผู้เข้าร่วมการวิจัย

**ชื่อโครงการ** ผลของการใช้แชตบอตปัญญาประดิษฐ์ต่อความรู้และพฤติกรรม
การป้องกันโรคไข้เลือดออกของประชาชนในชุมชน

**สิ่งที่ท่านต้องทำ**
1. ตอบแบบวัดความรู้และพฤติกรรมก่อนใช้งาน ประมาณ 10 นาที
2. ใช้แชตบอตเรียนรู้ อย่างน้อย 5 คำถาม
3. ตอบแบบวัดชุดเดิมอีกครั้ง และประเมินความพึงพอใจ

**การคุ้มครองข้อมูลส่วนบุคคล**
- ไม่เก็บชื่อ นามสกุล เลขบัตรประชาชน เบอร์โทร หรือที่อยู่โดยละเอียด
- ใช้รหัสสุ่มแทนตัวท่าน และรายงานผลเป็นภาพรวมเท่านั้น
- ท่านถอนตัวได้ทุกเมื่อโดยไม่มีผลกระทบต่อสิทธิการรับบริการใด ๆ

**ข้อจำกัดที่ต้องทราบ**
แชตบอตนี้ให้ความรู้ด้านสุขศึกษาเท่านั้น **ไม่วินิจฉัยโรคและไม่สั่งการรักษา**
หากมีอาการผิดปกติ กรุณาพบแพทย์ หรือโทร 1669 กรณีฉุกเฉิน
""")
        ok = st.checkbox("ข้าพเจ้าอ่านเข้าใจแล้ว และยินยอมเข้าร่วมการวิจัยด้วยความสมัครใจ")
        if st.button("เริ่มต้นใช้งาน", type="primary",
                     use_container_width=True, disabled=not ok):
            set_pid(db.new_pid())
            goto("profile")

    with tab2:
        st.info("หากเคยเริ่มไว้แล้ว ให้กรอกรหัสเดิมเพื่อทำต่อจากจุดที่ค้างไว้")
        code = st.text_input("รหัสผู้เข้าร่วม", placeholder="เช่น DF4K7QA").strip().upper()
        if st.button("ทำต่อ", use_container_width=True):
            if db.participant_exists(code):
                set_pid(code)
                st.rerun()
            else:
                st.error("ไม่พบรหัสนี้ในระบบ กรุณาตรวจสอบอีกครั้ง")


# ══════════════════════ ข้อมูลทั่วไป ══════════════════════
def page_profile(pid):
    st.header("ส่วนที่ 1 · ข้อมูลทั่วไป")
    st.success(f"**รหัสของท่านคือ `{pid}`** กรุณาจดไว้ หากปิดหน้าเว็บจะใช้รหัสนี้ทำต่อได้")

    with st.form("profile"):
        c1, c2 = st.columns(2)
        sex = c1.selectbox("เพศ", ["หญิง", "ชาย", "ไม่ระบุ"])
        age = c2.selectbox("ช่วงอายุ",
                           ["18-29 ปี", "30-44 ปี", "45-59 ปี", "60 ปีขึ้นไป"])
        edu = c1.selectbox("ระดับการศึกษาสูงสุด",
                           ["ประถมศึกษาหรือต่ำกว่า", "มัธยมศึกษา",
                            "อนุปริญญา/ปวส.", "ปริญญาตรีขึ้นไป"])
        role = c2.selectbox("บทบาทในชุมชน",
                            ["ประชาชนทั่วไป", "อสม.", "ผู้นำชุมชน", "อื่น ๆ"])
        village = c1.text_input("หมู่ที่ / ชุมชน (ไม่บังคับ)", placeholder="เช่น หมู่ 4")
        grp = c2.selectbox("กลุ่มทดลอง (ผู้วิจัยกำหนด)",
                           ["กลุ่มทดลอง", "กลุ่มควบคุม", "ไม่ระบุ"], index=0)
        exp = st.radio("ท่านหรือคนในครอบครัวเคยป่วยเป็นไข้เลือดออกหรือไม่",
                       ["ไม่เคย", "เคย", "ไม่แน่ใจ"], horizontal=True)

        if st.form_submit_button("บันทึกและไปต่อ", type="primary",
                                 use_container_width=True):
            db.save_profile(pid, {"sex": sex, "age_group": age, "education": edu,
                                  "role": role, "experience": exp,
                                  "village": village or None, "group_code": grp})
            st.rerun()


# ══════════════════════ แบบวัด (ใช้ร่วมกัน pre/post) ══════════════════════
def page_test(pid, phase):
    title = ("ส่วนที่ 2 · แบบวัดก่อนใช้งาน" if phase == "pre"
             else "ส่วนที่ 4 · แบบวัดหลังใช้งาน")
    st.header(title)
    if phase == "post":
        st.info("กรุณาตอบตามความเข้าใจและการปฏิบัติจริงในปัจจุบัน "
                "ไม่ต้องพยายามจำคำตอบเดิม")

    with st.form(f"test_{phase}"):
        st.subheader("ตอนที่ 1 · ความรู้เรื่องโรคไข้เลือดออก")
        st.caption("โปรดเลือกว่าข้อความต่อไปนี้ ถูก หรือ ผิด")
        ka = {}
        for code, text, _ in KNOWLEDGE_ITEMS:
            pick = st.radio(f"**{code}.** {text}", ["ถูก", "ผิด"],
                            horizontal=True, index=None, key=f"{phase}{code}")
            ka[code] = None if pick is None else (pick == "ถูก")

        st.divider()
        st.subheader("ตอนที่ 2 · พฤติกรรมการป้องกันโรค")
        st.caption("ในรอบ 1 เดือนที่ผ่านมา ท่านปฏิบัติสิ่งต่อไปนี้บ่อยเพียงใด")
        ba = {}
        for code, text in BEHAVIOR_ITEMS:
            pick = st.radio(f"**{code}.** {text}", list(BEHAVIOR_SCALE.keys()),
                            index=None, key=f"{phase}{code}")
            ba[code] = BEHAVIOR_SCALE.get(pick)

        submitted = st.form_submit_button("บันทึกคำตอบ", type="primary",
                                          use_container_width=True)

    if submitted:
        missing = ([c for c, v in ka.items() if v is None]
                   + [c for c, v in ba.items() if v is None])
        if missing:
            st.error(f"ยังตอบไม่ครบ {len(missing)} ข้อ ได้แก่ {', '.join(missing[:8])}"
                     + (" ..." if len(missing) > 8 else ""))
            return
        k_score = sum(1 for c, _, key in KNOWLEDGE_ITEMS if ka[c] == key)
        b_total = sum(ba.values())
        b_mean = round(b_total / len(ba), 2)
        db.save_test(pid, phase, k_score, b_total, b_mean, {**ka, **ba})
        st.rerun()


# ══════════════════════ หน้าแชต ══════════════════════
def render_answer(text, extra=None):
    st.markdown(text)
    if extra:
        st.caption("💬 คำถามที่เกี่ยวข้อง: " + " · ".join(f"*{e['topic']}*" for e in extra))
    st.caption(f"⚠️ {kb.DISCLAIMER}")


def page_chat(pid, can_finish):
    st.header("ส่วนที่ 3 · พูดคุยกับน้องลายเดียว")

    n = db.chat_count(pid)
    st.progress(min(n / MIN_QUESTIONS, 1.0),
                text=f"ถามแล้ว {n} คำถาม (ต้องการอย่างน้อย {MIN_QUESTIONS} คำถาม)")

    if "messages" not in S:
        S.messages = [{"role": "assistant", "content": kb.GREETING, "extra": None}]

    for m in S.messages:
        with st.chat_message(m["role"], avatar="🦟" if m["role"] == "assistant" else "🧑"):
            if m["role"] == "assistant":
                render_answer(m["content"], m.get("extra"))
            else:
                st.markdown(m["content"])

    st.write("**หัวข้อยอดนิยม กดเพื่อถามได้เลย**")
    quick = [("อาการเป็นยังไง", "symptom"), ("3 เก็บคืออะไร", "3keb"),
             ("ห้ามกินยาอะไร", "medicine"), ("ลูกน้ำอยู่ตรงไหน", "breeding"),
             ("แบบไหนต้องไป รพ.", "danger"), ("ความเชื่อผิด ๆ", "myth")]
    cols = st.columns(3)
    clicked = None
    for i, (label, _) in enumerate(quick):
        if cols[i % 3].button(label, use_container_width=True, key=f"q{i}"):
            clicked = label

    user_q = st.chat_input("พิมพ์คำถามของท่านที่นี่...") or clicked

    if user_q:
        S.messages.append({"role": "user", "content": user_q})

        if kb.check_emergency(user_q):
            ans, tid, topic, src, score = kb.EMERGENCY_REPLY, "emergency", "อาการฉุกเฉิน", "triage", 1.0
            extra = None
        else:
            hit, score = kb.match(user_q)
            if hit:
                ans, tid, topic, src = hit["a"], hit["id"], hit["topic"], "kb"
                extra = kb.related(hit["id"])
            else:
                ans = ask_llm_optional(user_q)
                tid, topic, src, extra = "unmatched", "ไม่พบหัวข้อ", "fallback", None

        S.messages.append({"role": "assistant", "content": ans, "extra": extra})
        db.save_chat(pid, n + 1, user_q, tid, topic, src, round(score, 3))
        st.rerun()

    st.divider()
    if can_finish:
        st.success("ท่านถามครบตามเกณฑ์แล้ว สามารถทำแบบวัดหลังใช้งานได้")
        if st.button("✅ ทำแบบวัดหลังใช้งาน", type="primary", use_container_width=True):
            goto("post")
    else:
        st.warning(f"กรุณาถามอีก {MIN_QUESTIONS - n} คำถาม "
                   "เพื่อให้ได้รับความรู้อย่างเพียงพอก่อนทำแบบวัด")


def ask_llm_optional(question: str) -> str:
    """เรียก LLM เฉพาะเมื่อผู้วิจัยตั้งค่า API key ไว้เท่านั้น"""
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not key:
        return kb.FALLBACK
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        sys = (
            "คุณคือ 'น้องลายเดียว' ผู้ช่วยสุขศึกษาเรื่องโรคไข้เลือดออกสำหรับชาวบ้านไทย\n"
            "กติกา\n"
            "1. ตอบเฉพาะเรื่องไข้เลือดออกและยุงลาย นอกเหนือจากนี้ให้ปฏิเสธอย่างสุภาพ\n"
            "2. ห้ามวินิจฉัยโรค ห้ามสั่งยา ห้ามระบุขนาดยาเฉพาะบุคคล\n"
            "3. ใช้ภาษาไทยง่าย ประโยคสั้น เหมาะกับผู้สูงอายุ ไม่เกิน 6 บรรทัด\n"
            "4. หากผู้ใช้เล่าอาการน่ากังวล ให้แนะนำพบแพทย์หรือโทร 1669 ทันที\n"
            "5. ย้ำเสมอว่าห้ามใช้แอสไพรินและ NSAIDs\n"
            "6. อ้างอิงแนวทางกรมควบคุมโรค กระทรวงสาธารณสุข"
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.3, max_tokens=400,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": question}])
        return r.choices[0].message.content
    except Exception:
        return kb.FALLBACK


# ══════════════════════ ความพึงพอใจ ══════════════════════
def page_satisfaction(pid):
    st.header("ส่วนที่ 5 · ความพึงพอใจต่อการใช้งาน")
    with st.form("sat"):
        sc = {}
        opts = list(LIKERT5.keys())[::-1]
        for code, text in SATISFACTION_ITEMS:
            pick = st.select_slider(f"**{code}.** {text}", options=opts,
                                    value="ปานกลาง", key=f"sat{code}")
            sc[code] = LIKERT5[pick]
        cm = st.text_area("ข้อเสนอแนะเพิ่มเติม (ไม่บังคับ)",
                          placeholder="เช่น อยากให้เพิ่มเรื่อง...")
        if st.form_submit_button("ส่งแบบประเมินและจบการทดลอง",
                                 type="primary", use_container_width=True):
            db.save_satisfaction(pid, sc, round(sum(sc.values()) / len(sc), 2),
                                 cm.replace("\n", " ").strip())
            st.rerun()


# ══════════════════════ หน้าสรุปผล ══════════════════════
def page_done(pid):
    r = db.get_result(pid)
    pre, post = r.get("pre"), r.get("post")
    st.balloons()
    st.success("บันทึกข้อมูลเรียบร้อยแล้ว ขอบคุณที่ร่วมเป็นส่วนหนึ่งของงานวิจัยครับ 🙏")

    st.subheader("สรุปผลของท่าน")
    c1, c2 = st.columns(2)
    c1.metric(f"คะแนนความรู้ (เต็ม {len(KNOWLEDGE_ITEMS)})",
              post["k_score"], f"{post['k_score'] - pre['k_score']:+d}")
    c2.metric("คะแนนพฤติกรรมเฉลี่ย (เต็ม 3)",
              post["b_mean"], f"{post['b_mean'] - pre['b_mean']:+.2f}")

    st.write(f"- ระดับความรู้หลังใช้งาน: **{interpret_k(post['k_score'])}**")
    st.write(f"- ระดับพฤติกรรมหลังใช้งาน: **{interpret_b(post['b_mean'])}**")
    st.write(f"- จำนวนคำถามที่ถาม: **{r['n_chat']}** คำถาม")

    st.divider()
    st.markdown("""
### 📌 สิ่งที่อยากให้ท่านทำต่อจากนี้
- สำรวจและกำจัดแหล่งเพาะพันธุ์ยุงลายรอบบ้าน **ทุก 7 วัน**
- ขัดผนังภาชนะทุกครั้ง ไม่ใช่เพียงเทน้ำทิ้ง
- ชวนเพื่อนบ้านทำพร้อมกัน เพราะยุงลายบินได้ไกลเพียง 30-50 เมตร

☎️ สายด่วนกรมควบคุมโรค **1422** | ฉุกเฉิน **1669**
""")
    st.info(f"รหัสของท่านคือ `{pid}` หากผู้วิจัยขอข้อมูลเพิ่มเติมให้แจ้งรหัสนี้")


# ══════════════════════ ส่วนผู้วิจัย ══════════════════════
def page_admin():
    st.header("🔐 แดชบอร์ดผู้วิจัย")
    correct = st.secrets.get("ADMIN_PW", os.getenv("ADMIN_PW", ""))
    if not correct:
        st.error("ยังไม่ได้ตั้งค่า ADMIN_PW ใน .streamlit/secrets.toml")
        return
    pw = st.text_input("รหัสผ่าน", type="password")
    if pw != correct:
        if pw:
            st.error("รหัสผ่านไม่ถูกต้อง")
        return

    df, chats = db.export_wide()
    if df.empty:
        st.warning("ยังไม่มีข้อมูลผู้เข้าร่วม")
        return

    done = df.dropna(subset=[c for c in ("pre_k_score", "post_k_score") if c in df])

    c1, c2, c3 = st.columns(3)
    c1.metric("ลงทะเบียนทั้งหมด", len(df))
    c2.metric("ทำครบทั้ง 2 ครั้ง", len(done))
    c3.metric("คำถามทั้งหมด", len(chats))

    if len(done) >= 2:
        st.subheader("ผลการวิเคราะห์ Paired t-test")
        rows = []
        for label, a, b in [("ความรู้", "pre_k_score", "post_k_score"),
                            ("พฤติกรรม (ค่าเฉลี่ย)", "pre_b_mean", "post_b_mean")]:
            res = paired_ttest(done[a].tolist(), done[b].tolist())
            if "error" in res:
                continue
            rows.append({
                "ตัวแปร": label, "n": res["n"],
                "ก่อน": round(res["mean_pre"], 2),
                "หลัง": round(res["mean_post"], 2),
                "ผลต่าง": round(res["mean_diff"], 2),
                "t": round(res["t"], 3), "df": res["df"],
                "p-value": f"{res['p']:.4f}",
                "นัยสำคัญ": "มี (p<.05)" if res["p"] < .05 else "ไม่มี",
                "Cohen's d": round(res["d"], 2),
                "ขนาดอิทธิพล": effect_label(res["d"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("ค่าความเชื่อมั่นของเครื่องมือ (Pre-test)")
        kcols = [f"pre_{c}" for c, _, _ in KNOWLEDGE_ITEMS if f"pre_{c}" in done]
        bcols = [f"pre_{c}" for c, _ in BEHAVIOR_ITEMS if f"pre_{c}" in done]
        cc1, cc2 = st.columns(2)
        if kcols:
            key = {c: k for c, _, k in KNOWLEDGE_ITEMS}
            mat = [[1 if row[f"pre_{c}"] == key[c] else 0
                    for c, _, _ in KNOWLEDGE_ITEMS if f"pre_{c}" in done]
                   for _, row in done.iterrows()]
            cc1.metric("KR-20 (แบบวัดความรู้)", f"{cronbach_alpha(mat):.3f}")
        if bcols:
            mat = done[bcols].values.tolist()
            cc2.metric("Cronbach's α (พฤติกรรม)", f"{cronbach_alpha(mat):.3f}")

    st.subheader("ข้อมูลรายบุคคล")
    st.dataframe(df, use_container_width=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ ข้อมูลวิจัย (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       f"dengue_results_{ts}.csv", "text/csv",
                       use_container_width=True)
    d2.download_button("⬇️ บันทึกการสนทนา (CSV)",
                       chats.to_csv(index=False).encode("utf-8-sig"),
                       f"dengue_chats_{ts}.csv", "text/csv",
                       use_container_width=True)

    if not chats.empty:
        st.subheader("หัวข้อที่ถูกถามมากที่สุด")
        st.bar_chart(chats["topic"].value_counts())
        miss = chats[chats.source == "fallback"]
        if not miss.empty:
            st.subheader(f"⚠️ คำถามที่ระบบตอบไม่ได้ ({len(miss)} ครั้ง)")
            st.caption("นำไปเพิ่มใน kb.py เพื่อพัฒนาระบบให้ดีขึ้น")
            st.dataframe(miss[["created_at", "question"]],
                         use_container_width=True, hide_index=True)


# ══════════════════════ ตัวควบคุมหลัก ══════════════════════
with st.sidebar:
    st.markdown(f"### 🦟 {APP_TITLE}")
    if current_pid():
        st.caption(f"รหัส `{S.pid}`")
        if st.button("ออกจากระบบ", use_container_width=True):
            for k in ("pid", "messages", "override_stage"):
                S.pop(k, None)
            st.query_params.clear()
            st.rerun()
    st.divider()
    admin_mode = st.checkbox("โหมดผู้วิจัย")
    st.divider()
    st.caption("☎️ ฉุกเฉิน 1669")
    st.caption("☎️ กรมควบคุมโรค 1422")

if admin_mode:
    page_admin()
else:
    pid = current_pid()
    if not pid:
        page_landing()
    else:
        stage = resolve_stage(pid)
        steps = {"profile": 1, "pre": 2, "chat": 3, "chat_done": 3,
                 "post": 4, "sat": 5, "done": 6}
        if stage in steps:
            st.progress(steps[stage] / 6,
                        text=f"ขั้นตอนที่ {steps[stage]} จาก 6")
        if stage == "profile":
            page_profile(pid)
        elif stage == "pre":
            page_test(pid, "pre")
        elif stage in ("chat", "chat_done"):
            page_chat(pid, can_finish=(stage == "chat_done"))
        elif stage == "post":
            page_test(pid, "post")
        elif stage == "sat":
            page_satisfaction(pid)
        else:
            page_done(pid)
