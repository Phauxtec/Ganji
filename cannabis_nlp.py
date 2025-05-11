import re

# --- Cannabis Keyword Sets ---
CANNABIS_KEYWORDS = set([
    "thc", "cbd", "cbn", "cbg", "thcv", "thca", "cbda", "cbc", "delta8", "delta9", "h4cbd", "hhc", "hhcp", "phc",
    "terpenes", "terpene", "terps", "myrcene", "limonene", "linalool", "caryophyllene", "pinene", "humulene",
    "ocimene", "terpinolene", "nerolidol", "bisabolol", "valencene", "indica", "sativa", "hybrid", "strain",
    "cultivar", "pheno", "genotype", "chemotype", "kush", "haze", "diesel", "purple", "cookie", "gelato", "runtz",
    "zkittlez", "mimosa", "gmo", "og", "flower", "bud", "nug", "trim", "shake", "pre-roll", "preroll", "joint",
    "blunt", "spliff", "bong", "pipe", "bowl", "rig", "dab", "cart", "cartridge", "vape", "disposable", "battery",
    "e-rig", "edible", "gummy", "lozenge", "tincture", "capsule", "softgel", "infused", "syrup", "beverage",
    "drinkable", "nano", "sublingual", "roa", "live resin", "live rosin", "sauce", "diamonds", "badder", "budder",
    "crumble", "wax", "shatter", "distillate", "isolate", "hash", "bubble hash", "kief", "high", "stoned", "uplifted",
    "euphoric", "couch lock", "mellow", "body buzz", "clear headed", "giggly", "focused", "introspective",
    "creative", "stimulating", "psychoactive", "anxiety", "insomnia", "inflammation", "chronic pain", "arthritis",
    "ptsd", "epilepsy", "nausea", "depression", "glaucoma", "fibromyalgia", "parkinson", "chemotherapy", "menu",
    "dispensary", "order", "delivery", "pickup", "inventory", "testing", "certificate", "coa", "compliance",
    "label", "batch", "harvest date", "packaged date", "weed", "zaza", "loud", "gas", "top shelf", "mid", "regs",
    "dope", "fire", "exotics", "flame", "stash", "plug", "trap", "420", "710", "lit", "baked", "blazed", "rolling",
    "hotbox", "rip", "session"
])

CANNABIS_PHRASES = [
    "get high", "feel relaxed", "recommend a strain", "strongest edible", "indica for sleep",
    "sativa for focus", "low thc high cbd", "terpene profile", "strain effects", "best gummy",
    "dab temp", "rosin press", "how to roll", "how to dab", "fast acting", "sublingual tincture",
    "how many mg", "med card", "what's in your pre rolls", "entourage effect", "clean lab tested",
    "i need a vape", "full spectrum", "broad spectrum", "nano emulsified", "legal limit", "first time discount"
]

# --- Match logic ---
def is_cannabis_related(text: str) -> bool:
    text = text.lower()
    words = set(re.findall(r"\b\w+\b", text))

    if words & CANNABIS_KEYWORDS:
        print(f"[CannabisCheck] ✅ Matched keywords: {', '.join(words & CANNABIS_KEYWORDS)}")
        return True

    for phrase in CANNABIS_PHRASES:
        if phrase in text:
            print(f"[CannabisCheck] ✅ Matched phrase: '{phrase}'")
            return True

    print("[CannabisCheck] ❌ No cannabis match")
    return False

# --- Inference logic ---
def infer_marijuana_context(message: str) -> str:
    text = message.lower().strip()

    if is_cannabis_related(text):
        return message

    if "pain" in text or "back" in text:
        return "What cannabis helps with pain?"
    if "sleep" in text or "insomnia" in text:
        return "What cannabis helps with sleep?"
    if "anxious" in text or "stress" in text:
        return "What cannabis helps with anxiety?"
    if "focus" in text or "brain fog" in text:
        return "What cannabis helps with focus?"
    if "recommend" in text or "suggest" in text:
        return "Can you recommend a cannabis strain?"
    if "vape" in text and "flower" in text:
        return "How does vaping differ from smoking flower?"
    
    return f"What cannabis product or service would help if someone said: '{message}'"
