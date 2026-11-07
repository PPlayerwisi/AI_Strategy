import os
import json

LOCALES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")
_LOCALE_CACHE = {}

def normalize_lang(lang: str) -> str:
    if not lang:
        return "en"
    l_clean = str(lang).strip().lower()
    if l_clean in ("uk", "ua", "ukr"):
        return "uk"
    return "en"

def load_locale(lang: str) -> dict:
    norm = normalize_lang(lang)
    if norm in _LOCALE_CACHE:
        return _LOCALE_CACHE[norm]
    path = os.path.join(LOCALES_DIR, f"{norm}.json")
    if not os.path.exists(path):
        path = os.path.join(LOCALES_DIR, "en.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _LOCALE_CACHE[norm] = data
    return data

def get_text(lang: str, key: str, **kwargs) -> str:
    data = load_locale(lang)
    messages = data.get("messages", {})
    template = messages.get(key)
    if template is None:
        fallback = load_locale("en").get("messages", {})
        template = fallback.get(key, key)
    return template.format(**kwargs) if kwargs else template

def get_unit_name(u_type: str, lang: str = "en") -> str:
    data = load_locale(lang)
    units = data.get("units", {})
    return units.get(str(u_type), str(u_type))

def get_struct_name(st_type: int, lang: str = "en") -> str:
    data = load_locale(lang)
    structs = data.get("structs", {})
    return structs.get(str(int(st_type)), str(st_type))

def get_ui_bundle(lang: str = "en") -> dict:
    data = load_locale(lang)
    return data.get("ui", {})