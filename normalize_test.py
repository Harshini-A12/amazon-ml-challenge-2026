import re
import unicodedata


def normalize_text(text):
    if not text:
        return ""

    # Convert accented characters to normal characters

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = "".join(
    " " if unicodedata.category(char).startswith("P") else char
    for char in text
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


examples = [
    "Payne Énterprises",
    "PAYNE ENTERPRISES LLC",
    "3315 FREMONT ST, PEORIA, IL",
    "6(29), C.I.T. Colony, 2ND Main Road Mylapore, Chennai, Tamil Nadu",
    "Maure Wilblims Colombier Inc",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "ஸ்ரீ சாய் எண்டர்பிரைசஸ்"
]


print("NORMALIZATION EXAMPLES")
print("=" * 60)

for example in examples:
    print("Original :", example)
    print("Normalized:", normalize_text(example))
    print()