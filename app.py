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
import json
import os

from anthropic import Anthropic
from ddgs import DDGS
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

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


@app.route("/translate", methods=["POST"])
def translate():
    """Receive a menu photo, ask Claude to read it, and return the results."""
    # 1. Grab the uploaded photo from the request.
    photo = request.files.get("photo")
    if photo is None:
        return jsonify({"error": "No photo was uploaded."}), 400

    # 2. Convert the photo into the text-based format the AI expects (base64).
    image_bytes = photo.read()
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = photo.mimetype or "image/jpeg"

    # 3. Send the photo + instructions to Claude.
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=4000,
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
    except Exception as error:  # noqa: BLE001 - show any problem plainly to the user
        return jsonify({"error": f"The AI request failed: {error}"}), 500

    # 4. Claude replies with JSON text. Sometimes it wraps the JSON in markdown
    #    ```json ... ``` fences, so we clean those off before reading it.
    reply_text = message.content[0].text.strip()
    if reply_text.startswith("```"):
        reply_text = reply_text.split("```")[1]  # take the part inside the fences
        if reply_text.startswith("json"):
            reply_text = reply_text[len("json"):]
        reply_text = reply_text.strip()

    # 5. Turn the text into real data and hand it back to the web page.
    try:
        data = json.loads(reply_text)
    except json.JSONDecodeError:
        return jsonify({"error": "The AI's reply was not in the expected format. Please try again."}), 500

    return jsonify(data)


@app.route("/images")
def images():
    """Find 2-3 web photos for a dish. The web page calls this for each dish,
    passing the dish name like /images?q=Margherita%20Pizza."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"images": []})

    try:
        # The web page sends an AI-cleaned dish name, so we search it directly.
        results = DDGS().images(query, max_results=4)
        urls = [r["image"] for r in results if r.get("image")][:3]
    except Exception:  # noqa: BLE001 - if search fails, just show no photos
        urls = []

    return jsonify({"images": urls})


if __name__ == "__main__":
    # host="0.0.0.0" lets your phone reach this server over your home wifi.
    app.run(host="0.0.0.0", port=5001, debug=True)
