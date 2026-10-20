import time

import streamlit as st
from deep_translator import GoogleTranslator, MyMemoryTranslator

st.set_page_config(page_title="Language Translator", page_icon="🌐")

LANGS = GoogleTranslator().get_supported_languages(as_dict=True)  # name -> code
NAMES = sorted(LANGS)
SRC_OPTIONS = ["auto-detect"] + NAMES
CODE_TO_NAME = {v.lower(): k for k, v in LANGS.items()}

# Google code (lowercase) -> NLLB code, used for the offline fallback (local PC only)
NLLB = {
    "en": "eng_Latn", "hi": "hin_Deva", "ml": "mal_Mlym", "ta": "tam_Taml",
    "te": "tel_Telu", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "kn": "kan_Knda", "pa": "pan_Guru", "ur": "urd_Arab", "fr": "fra_Latn",
    "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh-cn": "zho_Hans",
    "ar": "arb_Arab", "tr": "tur_Latn", "nl": "nld_Latn", "id": "ind_Latn",
    "vi": "vie_Latn", "th": "tha_Thai",
}


def detect_code(text):
    try:
        from langdetect import detect
        return detect(text).lower()
    except Exception:
        return "en"


@st.cache_resource(show_spinner="Loading offline model (first time only)...")
def load_nllb():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "facebook/nllb-200-distilled-600M"
    return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name)


def google(text, src_code, tgt_code):
    last = None
    for wait in (0, 2):
        time.sleep(wait)
        try:
            return GoogleTranslator(source=src_code, target=tgt_code).translate(text)
        except Exception as e:
            last = e
    raise last


def mymemory(text, src_name, tgt_name):
    # MyMemory needs an explicit source language, so detect it if needed
    if src_name == "auto-detect":
        src_name = CODE_TO_NAME.get(detect_code(text), "english")
    return MyMemoryTranslator(source=src_name, target=tgt_name).translate(text)


def offline(text, src_code, tgt_code):
    tgt = NLLB.get(tgt_code.lower())
    if not tgt:
        raise RuntimeError("offline model does not support this language")
    src = NLLB.get(src_code.lower()) if src_code != "auto" else None
    if not src:
        src = NLLB.get(detect_code(text), "eng_Latn")
    tok, model = load_nllb()
    tok.src_lang = src
    inputs = tok(text, return_tensors="pt")
    out = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_new_tokens=300,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def translate(text, src, tgt):
    src_code = "auto" if src == "auto-detect" else LANGS[src]
    tgt_code = LANGS[tgt]
    errors = []
    try:
        return google(text, src_code, tgt_code), "Google Translate"
    except Exception as e:
        errors.append(f"Google: {e}")
    try:
        return mymemory(text, src, tgt), "MyMemory"
    except Exception as e:
        errors.append(f"MyMemory: {e}")
    try:
        return offline(text, src_code, tgt_code), "offline model"
    except Exception as e:
        errors.append(f"Offline: {e}")
    raise RuntimeError(" | ".join(errors))


# ---------- UI ----------
st.session_state.setdefault("src", "auto-detect")
st.session_state.setdefault("tgt", "hindi")


def swap():
    s, t = st.session_state.src, st.session_state.tgt
    if s != "auto-detect":
        st.session_state.src, st.session_state.tgt = t, s


st.title("🌐 Language Translator")
st.caption("Pick the languages, type your text, and press Translate.")

c1, c2, c3 = st.columns([5, 1, 5])
c1.selectbox("From", SRC_OPTIONS, key="src")
c2.write("")
c2.button("⇄", on_click=swap, help="Swap languages")
c3.selectbox("To", NAMES, key="tgt")

text = st.text_area("Text to translate", height=160, max_chars=5000)

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please type some text first.")
    else:
        with st.spinner("Translating..."):
            try:
                result, how = translate(text.strip(), st.session_state.src, st.session_state.tgt)
                st.text_area("Translation", value=result, height=160)
                st.caption(f"Translated using {how}")
            except Exception as e:
                st.error(f"Translation failed. {e}")
