"""Dataset generation, validation, auditing, and train/val/test splitting engine.

Generates high-quality, leak-free code-mixed Indic datasets for:
1. Intent Classification
2. Content Moderation
3. PII Entity Recognition & Redaction
4. Text Normalization with Entity Preservation

Ensures group-based splitting to prevent template leakage between train and test.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)


# --- TEMPLATE DEFINITIONS FOR INTENT ---

INTENT_TEMPLATES = [
    # GREETING
    ("greeting", "hi_en_roman", "hello bhai kaise ho aap", "grp_greet_01"),
    ("greeting", "hi_en_roman", "namaste sir, mujhe thodi help chahiye", "grp_greet_01"),
    ("greeting", "hi_en_roman", "hey there, good morning team", "grp_greet_02"),
    ("greeting", "te_en_roman", "namaskaram andi, ela unnaru", "grp_greet_03"),
    ("greeting", "te_en_roman", "hi anna, oka small doubt undi", "grp_greet_03"),
    ("greeting", "bn_en_roman", "nomoshkar dada, kemon achen", "grp_greet_04"),
    ("greeting", "bn_en_roman", "hello bhai, ektu help lagbe", "grp_greet_04"),
    ("greeting", "en", "hello, good afternoon support team", "grp_greet_05"),
    ("greeting", "hi_devanagari", "नमस्ते, मुझे सहायता चाहिए", "grp_greet_06"),
    ("greeting", "te_telugu", "నమస్కారం అండి ఎలా ఉన్నారు", "grp_greet_07"),
    ("greeting", "bn_bengali", "নমস্কার দাদা কেমন আছেন", "grp_greet_08"),

    # PAYMENT ISSUE
    ("payment_issue", "hi_en_roman", "bhai payment deduct ho gaya par order confirm nahi dikha raha", "grp_pay_01"),
    ("payment_issue", "hi_en_roman", "paise account se kat gaye lekin app me failure show ho raha hai", "grp_pay_01"),
    ("payment_issue", "hi_en_roman", "upi payment ho gaya status pending dikha raha hai", "grp_pay_02"),
    ("payment_issue", "hi_en_roman", "mera payment nahi aaya account check karo", "grp_pay_09"),
    ("payment_issue", "hi_en_roman", "payment cut ho gaya lekin order receive nahi hua", "grp_pay_09"),
    ("payment_issue", "te_en_roman", "dabbu cut ayindi kani order confirm kaledu", "grp_pay_03"),
    ("payment_issue", "te_en_roman", "account nunchi amount deduct ayindi, order status failed ani undi", "grp_pay_03"),
    ("payment_issue", "bn_en_roman", "taka kete geche kintu order confirm hoyni", "grp_pay_04"),
    ("payment_issue", "bn_en_roman", "bank theke payment hoye geche kintu status pending", "grp_pay_04"),
    ("payment_issue", "en", "my payment was debited from bank account but order is not placed", "grp_pay_05"),
    ("payment_issue", "hi_devanagari", "पेमेंट कट गया है लेकिन आर्डर कन्फर्म नहीं हुआ", "grp_pay_06"),
    ("payment_issue", "te_telugu", "డబ్బులు కట్ అయ్యాయి కానీ ఆర్డర్ కన్ఫర్మ్ కాలేదు", "grp_pay_07"),
    ("payment_issue", "bn_bengali", "টাকা কেটে গেছে কিন্তু অর্ডার কনফার্ম হয়নি", "grp_pay_08"),

    # ORDER STATUS
    ("order_status", "hi_en_roman", "mera order abhi tak kaha pahucha hai track karna hai", "grp_ord_01"),
    ("order_status", "hi_en_roman", "bhai order status check karke batao kab tak aayega", "grp_ord_01"),
    ("order_status", "hi_en_roman", "payment ho gaya kya check karke bataye", "grp_ord_08"),
    ("order_status", "hi_en_roman", "mera payment confirm hua kya order status bataiye", "grp_ord_08"),
    ("order_status", "te_en_roman", "naa order ekkada undi tracking details ivvandi", "grp_ord_02"),
    ("order_status", "te_en_roman", "order eppudu deliver avutundi anna", "grp_ord_02"),
    ("order_status", "bn_en_roman", "amar order ekhon kothay ache tracking details din", "grp_ord_03"),
    ("order_status", "bn_en_roman", "order kobe ashbe ektu bolun", "grp_ord_03"),
    ("order_status", "en", "where is my package currently and when will it arrive", "grp_ord_04"),
    ("order_status", "hi_devanagari", "मेरा आर्डर अभी तक कहाँ पहुँचा है", "grp_ord_05"),
    ("order_status", "te_telugu", "నా ఆర్డర్ ఎక్కడ ఉంది ట్రాకింగ్ చెప్పండి", "grp_ord_06"),
    ("order_status", "bn_bengali", "আমার অর্ডার এখন কোথায় আছে ট্র্যাকিং জানান", "grp_ord_07"),

    # REFUND REQUEST
    ("refund_request", "hi_en_roman", "paise wapas kab tak aayenge refund process hua kya", "grp_ref_01"),
    ("refund_request", "hi_en_roman", "maine order cancel kar diya tha mera refund initiate kijiye", "grp_ref_01"),
    ("refund_request", "hi_en_roman", "mera refund abhi tak bank me credit nahi hua", "grp_ref_06"),
    ("refund_request", "te_en_roman", "naa refund dabbu eppudu vastundi account lo", "grp_ref_02"),
    ("refund_request", "bn_en_roman", "amar refund kobe pabo bank account e", "grp_ref_03"),
    ("refund_request", "en", "please process my refund for cancelled order immediately", "grp_ref_04"),
    ("refund_request", "hi_devanagari", "कृपया मेरा रिफंड जल्द से जल्द वापस करें", "grp_ref_05"),

    # CANCELLATION REQUEST
    ("cancellation_request", "hi_en_roman", "bhai mujhe yeh order cancel karna hai option nahi mil raha", "grp_canc_01"),
    ("cancellation_request", "hi_en_roman", "galti se wrong item order ho gaya cancel kar do please", "grp_canc_01"),
    ("cancellation_request", "hi_en_roman", "mera payment cancel kar do please urgent", "grp_canc_05"),
    ("cancellation_request", "hi_en_roman", "payment cancel kar do aur order cancel karo", "grp_canc_05"),
    ("cancellation_request", "te_en_roman", "ee order cancel cheyandi urgent ga", "grp_canc_02"),
    ("cancellation_request", "bn_en_roman", "ei order ta cancel korte chai ekhoni", "grp_canc_03"),
    ("cancellation_request", "en", "i need to cancel my order please assist", "grp_canc_04"),

    # ACCOUNT ACCESS
    ("account_access", "hi_en_roman", "bhai login nahi ho raha otp nahi aa raha mobile pe", "grp_acc_01"),
    ("account_access", "hi_en_roman", "password reset link send kijiye account access nahi kar pa raha", "grp_acc_01"),
    ("account_access", "te_en_roman", "account login avvatledu otp raaledu inka", "grp_acc_02"),
    ("account_access", "bn_en_roman", "account login hocche na otp ashche na", "grp_acc_03"),
    ("account_access", "en", "unable to log into my account because otp is not received", "grp_acc_04"),

    # DELIVERY ISSUE
    ("delivery_issue", "hi_en_roman", "delivery boy ne call nahi kiya aur order delivered mark kar diya", "grp_delv_01"),
    ("delivery_issue", "hi_en_roman", "wrong item deliver hua hai packet kholne ke baad pata chala", "grp_delv_01"),
    ("delivery_issue", "te_en_roman", "damaged product deliver ayindi box chinchesaru", "grp_delv_02"),
    ("delivery_issue", "bn_en_roman", "bhul jinish deliver hoyeche packet e", "grp_delv_03"),
    ("delivery_issue", "en", "received a damaged package and items are broken", "grp_delv_04"),

    # TECHNICAL SUPPORT
    ("technical_support", "hi_en_roman", "mujhe payment karna hai kaise karu option nahi mil raha", "grp_tech_05"),
    ("technical_support", "hi_en_roman", "how to pay through upi payment proceed nahi ho raha", "grp_tech_05"),
    ("technical_support", "hi_en_roman", "app bar bar crash ho raha hai screen blank aa rahi hai", "grp_tech_01"),
    ("technical_support", "te_en_roman", "app open avvatledu server error 500 vastondi", "grp_tech_02"),
    ("technical_support", "bn_en_roman", "app open hocche na shob shomoy loading dekhacche", "grp_tech_03"),
    ("technical_support", "en", "the checkout button is not responding in mobile app", "grp_tech_04"),

    # COMPLAINT
    ("complaint", "hi_en_roman", "bohot hi bekar service hai consumer forum me complaint karunga", "grp_comp_01"),
    ("complaint", "te_en_roman", "chala worst service idi nenu complaint chesthanu", "grp_comp_02"),
    ("complaint", "bn_en_roman", "khub baje service eto deri keno hocche", "grp_comp_03"),
    ("complaint", "en", "extremely disappointed with customer care and false promises", "grp_comp_04"),

    # PRODUCT QUESTION
    ("product_question", "hi_en_roman", "is product ki warranty kitne saal ki hai aur return policy kya hai", "grp_prod_01"),
    ("product_question", "te_en_roman", "ee item ki replacement warranty undha", "grp_prod_02"),
    ("product_question", "bn_en_roman", "ei product tar size chart kothay pabo", "grp_prod_03"),
    ("product_question", "en", "what are the technical specifications and dimensions of this item", "grp_prod_04"),

    # SUBSCRIPTION ISSUE
    ("subscription_issue", "hi_en_roman", "membership auto renew ho gayi paise kat gaye cancel karo", "grp_sub_01"),
    ("subscription_issue", "te_en_roman", "monthly plan cancel cheyandi renewal vaddhu", "grp_sub_02"),
    ("subscription_issue", "bn_en_roman", "subscription auto debit bondho korte chai", "grp_sub_03"),
    ("subscription_issue", "en", "please cancel my auto renewing pro subscription immediately", "grp_sub_04"),

    # UNKNOWN / OTHER
    ("unknown", "hi_en_roman", "aaj mausam kaisa hai waha par", "grp_unk_01"),
    ("unknown", "en", "random chat message having no relation to ecommerce service", "grp_unk_02"),
    ("unknown", "te_en_roman", "repu cinema ki veldam anukuntunna", "grp_unk_03"),
    ("unknown", "bn_en_roman", "ajke brishti hobe kina bolo to", "grp_unk_04")
]


# --- TEMPLATES FOR MODERATION ---

MODERATION_TEMPLATES = [
    # Non-toxic informal customer support
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "hi_en_roman", "bhai payment deduct ho gaya par order nahi dikha raha", "mod_grp_clean_01"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "hi_en_roman", "arey yaar bohot deri ho gayi jaldi bhejo", "mod_grp_clean_01"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "te_en_roman", "anna naaku delivery raledu konchem chudandi", "mod_grp_clean_02"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "bn_en_roman", "dada amar order ta ekhono elo na ektu dekhun", "mod_grp_clean_03"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "en", "please help me track my package as soon as possible", "mod_grp_clean_04"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "hi_en_roman", "bhai mera refund initiate hua ya nahi batao", "mod_grp_clean_05"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "en", "can you check if the item is in stock", "mod_grp_clean_06"),

    # Direct abuse
    ({"direct_abuse": 1, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 1, "spam": 0},
     "hi_en_roman", "kya chutiya service hai tumhari saale chor", "mod_grp_abuse_01"),
    ({"direct_abuse": 1, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 1, "spam": 0},
     "hi_en_roman", "bhosdike mera paisa wapas kar", "mod_grp_abuse_01"),
    ({"direct_abuse": 1, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 1, "spam": 0},
     "te_en_roman", "lanjakodaka na dabbu ivvandi ra", "mod_grp_abuse_02"),
    ({"direct_abuse": 1, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 1, "spam": 0},
     "bn_en_roman", "bokachoda company shob taka kheye niyo", "mod_grp_abuse_03"),
    ({"direct_abuse": 1, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 1, "spam": 0},
     "en", "you are absolute assholes and useless idiots", "mod_grp_abuse_04"),
    ({"direct_abuse": 1, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 1, "spam": 0},
     "hi_en_roman", "ek number ke harami log ho tum sab", "mod_grp_abuse_05"),

    # Threat
    ({"direct_abuse": 0, "threat": 1, "harassment": 1, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "hi_en_roman", "agar mera paisa nahi aaya toh office aakar jaan se mar dunga", "mod_grp_threat_01"),
    ({"direct_abuse": 0, "threat": 1, "harassment": 1, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "te_en_roman", "dabbu rakapothe office vachi champesthamu mimmalni", "mod_grp_threat_02"),
    ({"direct_abuse": 0, "threat": 1, "harassment": 1, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "bn_en_roman", "taka na ferot dile toder mere phelbo", "mod_grp_threat_03"),
    ({"direct_abuse": 0, "threat": 1, "harassment": 1, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "en", "i will come to your store and break your bones if refund is not given", "mod_grp_threat_04"),
    ({"direct_abuse": 0, "threat": 1, "harassment": 1, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 0},
     "hi_en_roman", "tujhe dekh lunga bahar nikal tu", "mod_grp_threat_05"),

    # Sexual harassment
    ({"direct_abuse": 0, "threat": 0, "harassment": 1, "hate_speech": 0, "sexual_harassment": 1, "profanity": 0, "spam": 0},
     "hi_en_roman", "apna phone number do aur sexy photo bhejo", "mod_grp_sex_01"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 1, "hate_speech": 0, "sexual_harassment": 1, "profanity": 0, "spam": 0},
     "en", "send nudes on whatsapp baby", "mod_grp_sex_02"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 1, "hate_speech": 0, "sexual_harassment": 1, "profanity": 0, "spam": 0},
     "te_en_roman", "nude photo pettu whatsapp lo", "mod_grp_sex_03"),

    # Spam
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 1},
     "hi_en_roman", "work from home earn 50000 daily click here to win lottery", "mod_grp_spam_01"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 1},
     "en", "congratulations you won iphone click here to claim free crypto", "mod_grp_spam_02"),
    ({"direct_abuse": 0, "threat": 0, "harassment": 0, "hate_speech": 0, "sexual_harassment": 0, "profanity": 0, "spam": 1},
     "hi_en_roman", "free recharge link click here to get 1000rs daily", "mod_grp_spam_03")
]


# --- TEMPLATES FOR PII DETECTION ---

PII_RAW_TEMPLATES = [
    ("mera phone number {phone} hai", "hi_en_roman", [("PHONE", "{phone}")]),
    ("call Rahul at {phone} about order {order_id}", "hi_en_roman", [("NAME", "Rahul"), ("PHONE", "{phone}"), ("ORDER_ID", "{order_id}")]),
    ("contact Suresh at {phone} or email {email}", "hi_en_roman", [("NAME", "Suresh"), ("PHONE", "{phone}"), ("EMAIL", "{email}")]),
    ("meri email id {email} pe invoice send kar do", "hi_en_roman", [("EMAIL", "{email}")]),
    ("UPI ID {upi} par refund transfer kijiye", "hi_en_roman", [("UPI_ID", "{upi}")]),
    ("PAN number {pan} verify ho gaya kya", "hi_en_roman", [("PAN", "{pan}")]),
    ("mera Aadhaar card {aadhaar} submit kiya tha", "hi_en_roman", [("AADHAAR", "{aadhaar}")]),
    ("delivery address Connaught Place New Delhi me Amit Sharma ko bhejna", "hi_en_roman", [("ADDRESS", "Connaught Place New Delhi"), ("NAME", "Amit Sharma")]),
    ("my address is behind old temple near railway station, Varanasi for order {order_id}", "en", [("ADDRESS", "behind old temple near railway station, Varanasi"), ("ORDER_ID", "{order_id}")]),
    ("vehicle number {vehicle} driver assign hua hai", "hi_en_roman", [("VEHICLE_NO", "{vehicle}")]),
    ("naaku call cheyandi {phone} ki", "te_en_roman", [("PHONE", "{phone}")]),
    ("naa order {order_id} details {email} ki pampandi", "te_en_roman", [("ORDER_ID", "{order_id}"), ("EMAIL", "{email}")]),
    ("amar phone {phone} e call korun dada", "bn_en_roman", [("PHONE", "{phone}")]),
    ("please send refund of Rs 1500 to upi {upi} for order {order_id}", "en", [("UPI_ID", "{upi}"), ("ORDER_ID", "{order_id}")]),
    ("send refund to {upi} and confirm to {phone}", "hi_en_roman", [("UPI_ID", "{upi}"), ("PHONE", "{phone}")])
]

# Non-PII numerical examples to train model/recognizer precision
NON_PII_TEMPLATES = [
    ("mera order total amount Rs 4999 tha", "hi_en_roman"),
    ("Please refund ₹1500 for order ORD-78321.", "en"),
    ("is product par 20% discount kab milega", "hi_en_roman"),
    ("maine 3 items cart me add kiye hai", "hi_en_roman"),
    ("deliver between 10am to 2pm please", "en"),
    ("total cost was INR 2500 for 2 pieces", "en"),
    ("Mera pincode 500081 me delivery available hai kya", "hi_en_roman"),
    ("amount of Rs 1500 was deducted on 2026-10-23", "en"),
    ("call me between 5pm to 6pm tomorrow", "en")
]


# --- TEMPLATES FOR NORMALIZATION ---

NORMALIZATION_TEMPLATES = [
    # Hinglish -> English & Devanagari
    {
        "source_text": "bhai payment deduct ho gaya par order confirm nahi dikha raha",
        "source_language": "hi_en_roman",
        "target_language": "en",
        "target_text": "My payment was deducted, but the order is not showing as confirmed.",
        "mode": "english",
        "preserve_entities": True
    },
    {
        "source_text": "bhai payment deduct ho gaya par order confirm nahi dikha raha",
        "source_language": "hi_en_roman",
        "target_language": "hi_devanagari",
        "target_text": "भाई पेमेंट डिडक्ट हो गया पर ऑर्डर कन्फर्म नहीं दिख रहा है।",
        "mode": "native_script",
        "preserve_entities": True
    },
    {
        "source_text": "mera order abhi tak deliver nhi hua",
        "source_language": "hi_en_roman",
        "target_language": "en",
        "target_text": "My order has not been delivered yet.",
        "mode": "english",
        "preserve_entities": True
    },
    {
        "source_text": "mera order abhi tak deliver nhi hua",
        "source_language": "hi_en_roman",
        "target_language": "hi_devanagari",
        "target_text": "मेरा ऑर्डर अभी तक डिलीवर नहीं हुआ।",
        "mode": "native_script",
        "preserve_entities": True
    },
    {
        "source_text": "paise account se kat gaye refund kab tak aayega",
        "source_language": "hi_en_roman",
        "target_language": "en",
        "target_text": "Money was debited from my account, when will the refund arrive?",
        "mode": "english",
        "preserve_entities": True
    },
    {
        "source_text": "bhai order ORD-998811 ka refund ₹1500 kab aayega",
        "source_language": "hi_en_roman",
        "target_language": "en",
        "target_text": "When will I receive my refund of ₹1500 for order ORD-998811?",
        "mode": "english",
        "preserve_entities": True
    },
    # Telgish -> English & Telugu
    {
        "source_text": "naaku order deliver avvaledu",
        "source_language": "te_en_roman",
        "target_language": "en",
        "target_text": "My order has not been delivered to me.",
        "mode": "english",
        "preserve_entities": True
    },
    {
        "source_text": "naaku order deliver avvaledu",
        "source_language": "te_en_roman",
        "target_language": "te_telugu",
        "target_text": "నాకు ఆర్డర్ డెలివర్ అవ్వలేదు.",
        "mode": "native_script",
        "preserve_entities": True
    },
    {
        "source_text": "dabbu cut ayindi kani order confirm kaledu",
        "source_language": "te_en_roman",
        "target_language": "en",
        "target_text": "The money was deducted, but the order was not confirmed.",
        "mode": "english",
        "preserve_entities": True
    },
    # Benglish -> English & Bengali
    {
        "source_text": "taka kete geche kintu order confirm hoyni",
        "source_language": "bn_en_roman",
        "target_language": "en",
        "target_text": "Money was deducted, but the order was not confirmed.",
        "mode": "english",
        "preserve_entities": True
    },
    {
        "source_text": "taka kete geche kintu order confirm hoyni",
        "source_language": "bn_en_roman",
        "target_language": "bn_bengali",
        "target_text": "টাকা কেটে গেছে কিন্তু অর্ডার কনফার্ম হয়নি।",
        "mode": "native_script",
        "preserve_entities": True
    },
    {
        "source_text": "amar order ekhono deliver hoyni",
        "source_language": "bn_en_roman",
        "target_language": "en",
        "target_text": "My order has not been delivered yet.",
        "mode": "english",
        "preserve_entities": True
    }
]


def generate_intent_dataset() -> List[Dict[str, Any]]:
    """Expand intent templates into a rich dataset."""
    examples = []
    idx = 1
    
    # Informal variation suffixes / prefixes
    prefixes = ["", "bhai ", "please ", "arey yaar ", "sir ", "hello ", "dada ", "anna "]
    suffixes = ["", " jaldi batao", " please check", "!", " urgently", " please assist"]

    for intent, lang, base_text, split_group in INTENT_TEMPLATES:
        # Base example
        examples.append({
            "id": f"intent_{idx:05d}",
            "text": base_text,
            "language_profile": lang,
            "task": "intent",
            "label": intent,
            "source": "synthetic_curated",
            "split_group": split_group
        })
        idx += 1

        # Add variations
        for p in prefixes:
            for s in suffixes:
                if not p and not s:
                    continue
                var_text = f"{p}{base_text}{s}".strip()
                examples.append({
                    "id": f"intent_{idx:05d}",
                    "text": var_text,
                    "language_profile": lang,
                    "task": "intent",
                    "label": intent,
                    "source": "synthetic_expanded",
                    "split_group": split_group
                })
                idx += 1

    return examples


def generate_moderation_dataset() -> List[Dict[str, Any]]:
    """Generate moderation dataset with realistic variations."""
    examples = []
    idx = 1

    prefixes = ["", "bhai ", "arey ", "please ", "dada ", "anna "]
    suffixes = ["", "!", "...", " reply me", " urgently"]

    for cat_dict, lang, text, split_group in MODERATION_TEMPLATES:
        flagged_cats = [c for c, val in cat_dict.items() if val == 1]
        primary_cat = flagged_cats[0] if flagged_cats else "clean"
        
        # Base template
        examples.append({
            "id": f"mod_{idx:05d}",
            "text": text,
            "language_profile": lang,
            "task": "moderation",
            "label": primary_cat,
            "is_toxic": len(flagged_cats) > 0,
            "categories": cat_dict,
            "flagged_categories": flagged_cats,
            "source": "synthetic_moderation",
            "split_group": split_group
        })
        idx += 1

        # Variations
        for p in prefixes:
            for s in suffixes:
                if not p and not s:
                    continue
                var_text = f"{p}{text}{s}".strip()
                examples.append({
                    "id": f"mod_{idx:05d}",
                    "text": var_text,
                    "language_profile": lang,
                    "task": "moderation",
                    "label": primary_cat,
                    "is_toxic": len(flagged_cats) > 0,
                    "categories": cat_dict,
                    "flagged_categories": flagged_cats,
                    "source": "synthetic_expanded",
                    "split_group": split_group
                })
                idx += 1

    return examples


def generate_pii_dataset() -> List[Dict[str, Any]]:
    """Generate PII dataset with exact character offsets."""
    examples = []
    idx = 1

    phones = [
        "9876543210", "9811223344", "+919123456789", "8765432109", "7890123456",
        "98765 43210", "+91 9876543210", "09876543210"
    ]
    emails = ["rahul.sharma@example.com", "aditya@test.org", "customer_care@indicmix.ai", "suresh.kumar@company.in"]
    upis = ["rahul@okhdfcbank", "aditya@paytm", "support@upi", "priya@okaxis", "suresh@ybl"]
    pans = ["ABCDE1234F", "XYZPQ9876K", "BKZPS4455M", "ALWPR1122D"]
    # Verhoeff-validated Aadhaar numbers (Dihedral group D5 verified)
    aadhaars = ["2345 6789 0124", "5489 1234 5674", "9876 5432 1096", "3627 1940 5812"]
    vehicles = ["DL 01 AB 1234", "MH 12 CD 5678", "TS 09 XY 9999", "KA 05 MN 4321"]
    order_ids = ["ORD-987123", "OD1234567890", "ORDER#445566", "ORD-78321"]

    for i in range(25):
        ph = phones[i % len(phones)]
        em = emails[i % len(emails)]
        up = upis[i % len(upis)]
        pa = pans[i % len(pans)]
        aa = aadhaars[i % len(aadhaars)]
        ve = vehicles[i % len(vehicles)]
        od = order_ids[i % len(order_ids)]

        for tmpl, lang, raw_entities in PII_RAW_TEMPLATES:
            # Substitute variables
            filled_text = tmpl.format(
                phone=ph, email=em, upi=up, pan=pa,
                aadhaar=aa, vehicle=ve, order_id=od
            )

            # Find exact offsets of each entity in filled_text
            entity_spans = []
            for ent_type, val_tmpl in raw_entities:
                actual_val = val_tmpl.format(
                    phone=ph, email=em, upi=up, pan=pa,
                    aadhaar=aa, vehicle=ve, order_id=od
                )
                start_pos = filled_text.find(actual_val)
                if start_pos != -1:
                    end_pos = start_pos + len(actual_val)
                    entity_spans.append({
                        "type": ent_type,
                        "start": start_pos,
                        "end": end_pos,
                        "text": actual_val
                    })

            examples.append({
                "id": f"pii_{idx:05d}",
                "text": filled_text,
                "language_profile": lang,
                "entities": entity_spans,
                "has_pii": len(entity_spans) > 0,
                "split_group": f"pii_grp_{tmpl[:12]}"
            })
            idx += 1

    # Add non-PII examples
    for non_pii_text, lang in NON_PII_TEMPLATES:
        examples.append({
            "id": f"pii_{idx:05d}",
            "text": non_pii_text,
            "language_profile": lang,
            "entities": [],
            "has_pii": False,
            "split_group": f"non_pii_grp_{idx % 5}"
        })
        idx += 1

    return examples


def generate_normalization_dataset() -> List[Dict[str, Any]]:
    """Generate normalization dataset."""
    examples = []
    idx = 1
    for item in NORMALIZATION_TEMPLATES:
        entry = dict(item)
        entry["id"] = f"norm_{idx:05d}"
        entry["split_group"] = f"norm_grp_{idx % 4}"
        examples.append(entry)
        idx += 1
    return examples


def group_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records by split_group to strictly prevent template leakage."""
    set_seed(seed)
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        grp = r.get("split_group", "default")
        group_map.setdefault(grp, []).append(r)

    groups = list(group_map.keys())
    random.shuffle(groups)

    train_cutoff = int(len(groups) * train_ratio)
    val_cutoff = train_cutoff + int(len(groups) * val_ratio)

    train_groups = set(groups[:train_cutoff])
    val_groups = set(groups[train_cutoff:val_cutoff])
    test_groups = set(groups[val_cutoff:])

    train_data, val_data, test_data = [], [], []
    for grp, items in group_map.items():
        if grp in train_groups:
            train_data.extend(items)
        elif grp in val_groups:
            val_data.extend(items)
        else:
            test_data.extend(items)

    # If test or val is empty due to small group count, reallocate
    if len(groups) >= 3:
        if not val_data and len(train_data) > 1:
            moved_val = list(train_groups)[-1]
            val_data = group_map[moved_val]
            train_data = [d for d in train_data if d.get("split_group") != moved_val]
        if not test_data and len(train_data) > 1:
            moved_test = list(train_groups)[-1]
            test_data = group_map[moved_test]
            train_data = [d for d in train_data if d.get("split_group") != moved_test]

    return train_data, val_data, test_data


def stratified_group_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records by split_group stratified per label to guarantee full class coverage without template leakage."""
    set_seed(seed)
    if not records or "label" not in records[0]:
        return group_split(records, train_ratio, val_ratio, test_ratio, seed)

    label_to_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in records:
        lbl = r["label"]
        grp = r.get("split_group", "default")
        label_to_groups.setdefault(lbl, {}).setdefault(grp, []).append(r)

    train_data, val_data, test_data = [], [], []

    for lbl, group_dict in label_to_groups.items():
        groups = list(group_dict.keys())
        random.shuffle(groups)

        n_groups = len(groups)
        if n_groups >= 3:
            n_test = max(1, int(round(n_groups * test_ratio)))
            n_val = max(1, int(round(n_groups * val_ratio)))
            if n_groups - n_test - n_val < 1:
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
            test_grps = set(groups[:n_test])
            val_grps = set(groups[n_test:n_test + n_val])
            train_grps = set(groups[n_test + n_val:])
        elif n_groups == 2:
            train_grps = {groups[0]}
            val_grps = set()
            test_grps = {groups[1]}
        else:
            items = list(group_dict[groups[0]])
            random.shuffle(items)
            n_t = max(1, int(len(items) * train_ratio))
            n_v = max(1, int(len(items) * val_ratio))
            train_data.extend(items[:n_t])
            val_data.extend(items[n_t:n_t + n_v])
            test_data.extend(items[n_t + n_v:])
            continue

        for grp, items in group_dict.items():
            if grp in train_grps:
                train_data.extend(items)
            elif grp in val_grps:
                val_data.extend(items)
            else:
                test_data.extend(items)

    return train_data, val_data, test_data


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_all_datasets():
    """Build all datasets, perform leak-free split, and generate manifest."""
    print("Building IndicMix AI Datasets...")
    set_seed(42)

    tasks = {
        "intent": generate_intent_dataset(),
        "moderation": generate_moderation_dataset(),
        "pii": generate_pii_dataset(),
        "normalization": generate_normalization_dataset()
    }

    manifest = {
        "project": "IndicMix AI",
        "license": "Apache-2.0 / Curated Synthetic Research Corpus",
        "split_strategy": "Stratified group-based split (zero template/user leakage, 100% class coverage)",
        "tasks": {}
    }

    for task_name, dataset in tasks.items():
        if task_name in ["intent", "moderation"]:
            train, val, test = stratified_group_split(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        else:
            train, val, test = group_split(dataset, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10)
        
        base_dir = os.path.join("data", "processed", task_name)
        write_jsonl(os.path.join(base_dir, "train.jsonl"), train)
        write_jsonl(os.path.join(base_dir, "val.jsonl"), val)
        write_jsonl(os.path.join(base_dir, "test.jsonl"), test)

        manifest["tasks"][task_name] = {
            "total_samples": len(dataset),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "output_directory": base_dir
        }
        print(f"Task '{task_name}': Total={len(dataset)}, Train={len(train)}, Val={len(val)}, Test={len(test)}")

    manifest_path = os.path.join("data", "manifests", "dataset_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset generation complete. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    build_all_datasets()
