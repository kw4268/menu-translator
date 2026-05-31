"""
Menu Translator — backend server.

What this file does, in plain English:
  1. Shows your dad a simple web page (he opens it on his phone).
  2. When he takes a photo of a menu, the photo is sent here.
  3. We hand the photo to Claude, who reads the dishes, translates them to
     Chinese, and estimates the nutrition.
  4. We send those results back to the page so he can read them.

You don't need to understand every line. The comments explain the important bits.
"""

import base64
import io
import json
import os
import urllib.parse
import urllib.request

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Lets us read iPhone "HEIC" photos. If the library isn't available, we simply
# skip it — common formats (JPEG/PNG) still work.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001
    pass

# Load the secret API key from the private ".env" file.
# override=True makes our .env file win over any empty/old key already in the system.
load_dotenv(override=True)

# This is the AI model we use. Sonnet is a great balance of smart + affordable.
# (You can change it later — e.g. to "claude-opus-4-8" for the very best quality,
#  or "claude-haiku-4-5-20251001" for the cheapest.)
MODEL = "claude-sonnet-4-6"

# template_folder="." means index.html sits right next to app.py (a simple,
# flat layout — no subfolders needed).
app = Flask(__name__, template_folder=".")
client = Anthropic()  # automatically reads ANTHROPIC_API_KEY from the environment

# These instructions tell Claude exactly what to do with the menu photo.
# The person reading the results eats clean and wants to AVOID processed foods,
# so we also ask Claude to give a clean-eating recommendation for each dish.
SYSTEM_PROMPT = """You are a friendly menu translator helping a Chinese-speaking
person understand a restaurant menu written in another language. This person eats
clean and wants to AVOID processed/ultra-processed foods, deep-fried items, and
heavy added sugar. They prefer whole, fresh, minimally-processed ingredients.

Look at the menu photo and find every dish (skip section headers, prices, and
non-food items). For EACH dish, provide:
  - "original": the dish name exactly as written on the menu
  - "chinese": the dish name translated into Simplified Chinese (简体中文)
  - "description": a short, simple description in Simplified Chinese of what the
     dish is and its main ingredients (one sentence)
  - "calories": your best ESTIMATE of calories per typical serving (a number only)
  - "protein_g", "carbs_g", "fat_g": estimated grams per serving (numbers only)
  - "rating": exactly one of these three strings, judging how well it fits a clean,
     unprocessed diet:
        "推荐"   (recommended: fresh, whole, minimally processed)
        "适中"   (okay in moderation)
        "不推荐" (avoid: processed, deep-fried, or heavy in sugar/refined ingredients)
  - "health_note": one short sentence in Simplified Chinese explaining the rating,
     focused on whether the dish is clean / how processed it is.
  - "image_query": a SHORT English search phrase that will find an accurate photo
     of this exact dish. Use the dish's standard, well-known name — NOT the menu's
     fancy or restaurant-specific wording. Drop brand names, chef names, and flair
     words ("Nonna's", "House Special", "Signature"). Add the key ingredient or
     cuisine if it helps identify it. Examples:
        menu "Nonna's Rigatoni alla Vodka"  ->  "rigatoni alla vodka pasta"
        menu "The Big Sur Burger"            ->  "beef cheeseburger"
        menu "Grandma's Garden Soup"         ->  "vegetable soup"

The nutrition numbers are friendly estimates, not exact. That is okay and expected.

Respond with ONLY a JSON object in this exact shape, and nothing else:
{
  "dishes": [
    {"original": "...", "chinese": "...", "description": "...",
     "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0,
     "rating": "推荐", "health_note": "...", "image_query": "..."}
  ]
}
If you cannot find any dishes, return {"dishes": []}."""


@app.route("/")
def home():
    """Serve the web page your dad sees."""
    return render_template("index.html")


def to_jpeg(raw_bytes):
    """Turn any uploaded photo (HEIC, PNG, huge phone photo, etc.) into a tidy
    JPEG that the AI can always read. Shrinks very large photos to speed things up."""
    img = Image.open(io.BytesIO(raw_bytes))
    img = img.convert("RGB")  # drop transparency / odd color modes
    # The AI sees no benefit beyond ~1600px, so shrink big photos for speed.
    max_side = 1600
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def parse_dishes(message):
    """Pull the JSON out of Claude's reply, tolerating markdown ``` fences or any
    stray text around it. Returns a dict, or None if it can't be parsed."""
    if not message.content:
        return None
    text = message.content[0].text.strip()
    # Remove markdown code fences if present.
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Try the text as-is, then just the {...} slice (handles extra prose around it).
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "dishes" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


@app.route("/translate", methods=["POST"])
def translate():
    """Receive a menu photo, ask Claude to read it, and return the results."""
    # 1. Grab the uploaded photo from the request.
    photo = request.files.get("photo")
    if photo is None:
        return jsonify({"error": "No photo was uploaded."}), 400

    # 2. Normalize the photo to a JPEG and encode it for the AI.
    try:
        jpeg_bytes = to_jpeg(photo.read())
    except Exception:  # noqa: BLE001
        return jsonify({"error": "无法读取这张照片，请重试或换一张。(Could not read that photo.)"}), 400
    image_data = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
    media_type = "image/jpeg"

    # 3. Ask Claude to read & translate the menu. max_tokens is generous so big
    #    menus don't get cut off. We try up to twice in case a reply comes back
    #    in an unparseable shape.
    def ask_claude():
        return client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": "Here is the menu. Please translate it."},
                    ],
                }
            ],
        )

    data = None
    for attempt in range(2):
        try:
            message = ask_claude()
        except Exception as error:  # noqa: BLE001 - show any problem plainly
            return jsonify({"error": f"The AI request failed: {error}"}), 500
        data = parse_dishes(message)
        if data is not None:
            break

    # 4. If we still couldn't read a clean reply, ask the user to retry.
    if data is None:
        return jsonify({"error": "菜单较复杂，没能完整读取，请重试或拍清晰一点的照片。"
                                 "(Couldn't read the menu cleanly — please try again.)"}), 500

    return jsonify(data)


def pexels_images(query):
    """Search Pexels for food photos. Reliable from servers and always real,
    safe photos. Returns a list of image URLs (empty if not configured/fails)."""
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        return []

    url = "https://api.pexels.com/v1/search?" + urllib.parse.urlencode({
        "query": query,
        "per_page": 3,
        "orientation": "landscape",
    })
    # Pexels sits behind Cloudflare, which blocks requests that don't look like a
    # browser — so we send a normal browser User-Agent along with the API key.
    request_obj = urllib.request.Request(url, headers={
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    })
    try:
        with urllib.request.urlopen(request_obj, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [p["src"]["large"] for p in data.get("photos", []) if p.get("src")]
    except Exception:  # noqa: BLE001
        return []


@app.route("/images")
def images():
    """Find up to 3 photos for a dish. The web page calls this for each dish,
    passing the dish name like /images?q=Margherita%20Pizza."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"images": []})

    # A safety backstop: never show images from these adult/unsafe domains.
    BLOCKED = ("xhcdn", "xhamster", "pornhub", "phncdn", "xvideos",
               "xnxx", "redtube", "porn", "nsfw", "adult")

    def is_safe(url):
        return url and not any(bad in url.lower() for bad in BLOCKED)

    # Photos come from Pexels (reliable, real, safe food photos). If there's no
    # match, the page shows a clean "no photo" placeholder — we never show
    # wrong/random images.
    urls = [u for u in pexels_images(query) if is_safe(u)][:3]

    return jsonify({"images": urls})


@app.errorhandler(Exception)
def handle_any_error(error):
    """Safety net: if anything unexpected goes wrong, reply with JSON (not an
    HTML error page) so the web page can always show a friendly message."""
    return jsonify({"error": f"出错了，请重试。(Unexpected error: {error})"}), 500


if __name__ == "__main__":
    # host="0.0.0.0" lets your phone reach this server over your home wifi.
    app.run(host="0.0.0.0", port=5001, debug=True)
