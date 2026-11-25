"""Language detection and reply templates for the offline FakeLLM."""

from __future__ import annotations

import re

SCRIPTS = {
    "hi": re.compile(r"[ऀ-ॿ]"),
    "bn": re.compile(r"[ঀ-৿]"),
    "ta": re.compile(r"[஀-௿]"),
}
HINGLISH = re.compile(
    r"\b(kya|hai|hain|mera|meri|mujhe|karna|karo|kab|kaise|kitne|din|nahi|chahiye|ho sakta|"
    r"wapas|paisa|kar do|bata)\b",
    re.IGNORECASE,
)

TEMPLATES: dict[str, dict[str, str]] = {
    "returns": {
        "hi": "आप डिलीवरी के 30 दिनों के भीतर सामान वापस कर सकते हैं। [returns]",
        "hi-Latn": "Aap delivery ke 30 din ke andar item return kar sakte hain. [returns]",
        "ta": "டெலிவரி ஆன 30 நாட்களுக்குள் பொருளைத் திருப்பி அனுப்பலாம். [returns]",
        "bn": "ডেলিভারির 30 দিনের মধ্যে পণ্য ফেরত দিতে পারেন। [returns]",
    },
    "shipping": {
        "hi": "₹499 से ऊपर के ऑर्डर पर शिपिंग मुफ़्त है। [shipping]",
        "hi-Latn": "₹499 se upar ke order par shipping free hai. [shipping]",
        "ta": "₹499க்கு மேல் உள்ள ஆர்டர்களுக்கு இலவச ஷிப்பிங். [shipping]",
        "bn": "₹499-এর বেশি অর্ডারে শিপিং বিনামূল্যে। [shipping]",
    },
    "cod": {
        "hi": "₹10,000 तक के ऑर्डर पर कैश ऑन डिलीवरी उपलब्ध है। [cod]",
        "hi-Latn": "₹10,000 tak ke order par cash on delivery milta hai. [cod]",
        "ta": "₹10,000 வரை உள்ள ஆர்டர்களுக்கு கேஷ் ஆன் டெலிவரி உண்டு. [cod]",
        "bn": "₹10,000 পর্যন্ত অর্ডারে ক্যাশ অন ডেলিভারি পাওয়া যায়। [cod]",
    },
    "refund-timeline": {
        "hi": "रिफ़ंड 5-7 कार्यदिवसों में मूल भुगतान माध्यम में आता है। [refund-timeline]",
        "hi-Latn": "Refund 5-7 business days mein original payment method mein aata hai. "
        "[refund-timeline]",
        "ta": "பணம் 5-7 வேலை நாட்களில் அசல் கட்டண முறைக்குத் திரும்பும். [refund-timeline]",
        "bn": "রিফান্ড 5-7 কার্যদিবসে মূল পেমেন্ট পদ্ধতিতে ফেরত আসে। [refund-timeline]",
    },
    "status": {
        "hi": "ऑर्डर {oid} ({amount}) की स्थिति: {status}।",
        "hi-Latn": "Aapke order {oid} ({amount}) ka status {status} hai.",
        "ta": "ஆர்டர் {oid} ({amount}) நிலை: {status}.",
        "bn": "অর্ডার {oid} ({amount})-এর অবস্থা: {status}।",
    },
    "cancelled": {
        "hi": "आपका ऑर्डर {oid} रद्द कर दिया गया है।",
        "hi-Latn": "Aapka order {oid} cancel ho gaya hai.",
        "ta": "உங்கள் ஆர்டர் {oid} ரத்து செய்யப்பட்டது.",
        "bn": "আপনার অর্ডার {oid} বাতিল করা হয়েছে।",
    },
    "not_found": {
        "hi": "मुझे यह जानकारी हमारी नीतियों में नहीं मिली। क्या मैं आपको किसी सहायक से जोड़ूँ?",
        "hi-Latn": "Mujhe yeh jaankari hamari policies mein nahi mili. Kya main aapko human "
        "agent se jod doon?",
        "ta": "இந்த தகவல் எங்கள் கொள்கைகளில் இல்லை. மனித உதவியாளரை இணைக்கட்டுமா?",
        "bn": "এই তথ্য আমাদের নীতিতে পাইনি। মানব সহায়কের সাথে যুক্ত করব?",
    },
    "refusal": {
        "hi": "माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता। मैं ऑर्डर, रिटर्न और शिपिंग में मदद कर सकता हूँ।",
        "hi-Latn": "Maaf kijiye, main ismein madad nahi kar sakta. Main orders, returns aur "
        "shipping mein madad kar sakta hoon.",
        "ta": "மன்னிக்கவும், இதற்கு என்னால் உதவ முடியாது.",
        "bn": "দুঃখিত, এতে আমি সাহায্য করতে পারব না।",
    },
}


def detect_language(text: str) -> str:
    """'en', 'hi' (Devanagari), 'hi-Latn' (Hinglish), 'ta' or 'bn'."""
    for lang, rx in SCRIPTS.items():
        if rx.search(text):
            return lang
    if len(HINGLISH.findall(text)) >= 2:
        return "hi-Latn"
    return "en"


def localise(key: str, lang: str, **kw: str) -> str | None:
    tmpl = TEMPLATES.get(key, {}).get(lang)
    return tmpl.format(**kw) if tmpl else None
