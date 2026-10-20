from deep_translator import GoogleTranslator

LANGS = GoogleTranslator().get_supported_languages(as_dict=True)  # name -> code
target = "hindi"

HELP = """Commands:
  /to <language>   set target language (e.g. /to french)
  /langs           list all supported languages
  /help            show this help
  (empty line)     quit
Just type any text in any language to translate it."""


def translate(text):
    return GoogleTranslator(source="auto", target=LANGS[target]).translate(text)


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
            print(f"[{target}] {translate(q)}")
        except Exception as e:
            print("Translation failed:", e)
