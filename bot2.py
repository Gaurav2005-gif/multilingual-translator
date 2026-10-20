import time
from deep_translator import GoogleTranslator

LANGS = GoogleTranslator().get_supported_languages(as_dict=True)  # name -> code
target = "hindi"

# Google code (lowercase) -> NLLB code, used for the offline fallback
NLLB = {
    "en": "eng_Latn", "hi": "hin_Deva", "ml": "mal_Mlym", "ta": "tam_Taml",
    "te": "tel_Telu", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "kn": "kan_Knda", "pa": "pan_Guru", "ur": "urd_Arab", "fr": "fra_Latn",
    "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh-cn": "zho_Hans",
    "ar": "arb_Arab", "tr": "tur_Latn", "nl": "nld_Latn", "id": "ind_Latn",
    "vi": "vie_Latn", "th": "tha_Thai",
}

HELP = """Commands:
  /to <language>   set target language (e.g. /to french)
  /langs           list all supported languages
  /help            show this help
  (empty line)     quit
Just type any text in any language to translate it."""

_cache = {}
_nllb = {}


def google(text, code):
    last = None
    for wait in (0, 2, 5):
        time.sleep(wait)
        try:
            return GoogleTranslator(source="auto", target=code).translate(text)
        except Exception as e:
            last = e
    raise last


def nllb(text, code):
    tgt = NLLB.get(code.lower())
    if not tgt:
        raise RuntimeError("offline model has no support for this language")
    if not _nllb:
        print("(Google is busy - loading offline model, one moment...)")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        name = "facebook/nllb-200-distilled-600M"
        _nllb["tok"] = AutoTokenizer.from_pretrained(name)
        _nllb["model"] = AutoModelForSeq2SeqLM.from_pretrained(name)
    tok, model = _nllb["tok"], _nllb["model"]
    src = "eng_Latn"
    try:
        from langdetect import detect
        src = NLLB.get(detect(text).lower(), "eng_Latn")
    except Exception:
        pass
    tok.src_lang = src
    inputs = tok(text, return_tensors="pt")
    out = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
        max_new_tokens=200,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def translate(text):
    code = LANGS[target]
    key = (text, code)
    if key in _cache:
        return _cache[key], "cache"
    try:
        result, how = google(text, code), "google"
    except Exception:
        result, how = nllb(text, code), "offline"
    _cache[key] = result
    return result, how


print("Translator bot ready. Target language:", target)
print(HELP)

while True:
    q = input("\n> ").strip()
    if not q:
        break
    if q == "/help":
        print(HELP)
    elif q == "/langs":
        print(", ".join(sorted(LANGS)))
    elif q.startswith("/to "):
        name = q[4:].strip().lower()
        if name in LANGS:
            target = name
            print("Target language set to", target)
        else:
            print("Unknown language. Type /langs to see the list.")
    else:
        try:
            out, how = translate(q)
            print(f"[{target} | {how}] {out}")
        except Exception as e:
            print("Translation failed:", e)
