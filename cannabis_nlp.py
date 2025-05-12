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

CANNABIS_STRAINS = set([
    "blue dream", "girl scout cookies", "sour diesel", "green crack", "northern lights", "granddaddy purple",
    "og kush", "white widow", "pineapple express", "gelato", "wedding cake", "gdp", "ak-47", "super lemon haze",
    "jack herer", "bubba kush", "cherry pie", "banana kush", "gorilla glue", "gg4", "purple haze", "chemdawg",
    "durban poison", "maui wowie", "zombie kush", "ice cream cake", "slurricane", "forbidden fruit", "zkittlez",
    "runtz", "mac", "mac1", "trainwreck", "cereal milk", "mimosa", "gmo", "do-si-dos", "skywalker og", "9lb hammer",
    "strawberry cough", "apple fritter", "platinum kush", "sherbert", "chocolope", "sundae driver", "animal mints",
    "tropicana cookies", "lemon skunk", "la confidential", "death star", "alien og", "sour og", "amnesia haze",
    "bubblegum", "critical mass", "hash plant", "candyland", "headband", "fire og", "ice", "blueberry",
    "lemon haze", "grape ape", "kandy kush", "peyote cookies", "blackberry kush", "purple punch", "sour tsunami",
    "agent orange", "critical kush", "orange bud", "peach rings", "lambs bread", "afghan kush", "blue cheese",
    "chem dawg", "god’s gift", "holy grail kush", "sour bubble", "tangie", "thin mint cookies", "strawberry banana",
    "nyc diesel", "super silver haze", "white fire og", "white rhino", "wappa", "ace of spades", "apple jacks",
    "sunset sherbet", "lava cake", "gelatti", "banana punch", "wookie", "zookies", "mochi", "lemon tree",
    "triple scoop", "biscotti", "motorbreath", "bacio gelato", "sour space candy", "afgoo", "grease monkey",
    "purple urkle", "tahoe og", "nuken", "violator kush", "pink kush", "wedding pie", "double dream"
])


CANNABIS_PHRASES = [
    "get high", "feel relaxed", "recommend a strain", "strongest edible", "indica for sleep",
    "sativa for focus", "low thc high cbd", "terpene profile", "strain effects", "best gummy",
    "dab temp", "rosin press", "how to roll", "how to dab", "fast acting", "sublingual tincture",
    "how many mg", "med card", "what's in your pre rolls", "entourage effect", "clean lab tested",
    "i need a vape", "full spectrum", "broad spectrum", "nano emulsified", "legal limit", "first time discount"
]

# --- Match logic ---
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

    for strain in CANNABIS_STRAINS:
        if strain in text:
            print(f"[CannabisCheck] ✅ Matched known strain: '{strain}'")
            return True

    print("[CannabisCheck] ❌ No cannabis match")
    return False

# --- Inference logic ---
def infer_marijuana_context(message: str) -> str:
    text = message.lower().strip()

    if is_cannabis_related(text):
        return message

    # === Symptom-based guidance ===
    if any(x in text for x in ["pain", "back", "arthritis", "cramps", "aches", "migraine"]):
        return "Are you looking for cannabis options that help with physical pain relief?"

    if any(x in text for x in ["sleep", "insomnia", "tired", "can't sleep", "restless"]):
        return "Looking for something to help with sleep or relaxation?"

    if any(x in text for x in ["anxious", "stress", "panic", "overthinking", "nerves"]):
        return "Are you trying to ease anxiety or manage stress with cannabis?"

    if any(x in text for x in ["focus", "adhd", "concentrate", "brain fog"]):
        return "Would you like a cannabis product that supports focus or mental clarity?"

    if any(x in text for x in ["mood", "depression", "low energy", "no motivation"]):
        return "Would an uplifting or mood-boosting cannabis strain help?"

    if any(x in text for x in ["hungry", "appetite", "nausea"]):
        return "Are you looking for something to help with appetite or nausea?"

    # === General help ===
    if "recommend" in text or "suggest" in text:
        return "Can you recommend a good cannabis strain or product for a beginner?"

    if "vape" in text and "flower" in text:
        return "How does vaping cannabis compare to smoking flower?"

    if any(x in text for x in ["first time", "never used", "new to this"]):
        return "It’s your first time? I can guide you through different cannabis options!"

    if any(x in text for x in ["don't know", "not sure", "what do you have", "what's good"]):
        return "Not sure where to start? Let me help you find a strain based on how you want to feel."

    return (
        f"I'm here to help with cannabis-related topics. Could you clarify if this is about pain relief, sleep, "
        f"focus, anxiety, or a specific product you're interested in?"
    )
