import asyncio
import base64
import keyboard
import mss
from google import genai
from google.genai import types
import io
import os
from PIL import Image

# Initialisation du client OpenAI
openai_api_key = "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"


# Fonction : capture une portion de l'écran (80% verticalement, centré)
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        width, height = monitor["width"], monitor["height"]

        # On capture 80% de la hauteur centrée
        crop_height = int(height * 0.8)
        top = (height - crop_height) // 2
        left, right = 0, width

        bbox = {"top": top, "left": left, "width": width, "height": crop_height}
        img = sct.grab(bbox)

        # Convertir en bytes
        img_pil = Image.frombytes("RGB", img.size, img.rgb)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")


# Sauvegarde l'image capturée (optionnel)
def save_image(base64_image, filename="capture.png"):
    img_data = base64.b64decode(base64_image)
    # Préciser le chemin dans de le dossier courant
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, "wb") as f:
        f.write(img_data)
    print(f"💾 Image sauvegardée sous {filename}")

def obtenir_reponse(img_b64):
    client = genai.Client(api_key='AIzaSyByCq2zoA95urIFXjvhYYZaIGl0BF8IWBQ')
    response = ""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Tu es un expert en mathématiques et en ingénierie informatique. "
                    "Analyse la capture d’écran contenant une question. "
                    "Réponds uniquement : "
                    "- par la ou les lettres/numéro du QCM (ex: A, B, 1, 2, 2 et 3, ou A et C), "
                    "- si le qcm n'a pas de numérotation ou de lettres, réponds par A, B, C, etc. "
                    "- si c'est une question ouverte, réponds par la réponse exacte. "
                    "- ou par la réponse exacte si aucune proposition n’est donnée."
                    "Si la question est illisible ou incomplète, réponds exactement : ERREUR. "
                    "Aucune explication ni texte supplémentaire."
                )
            ),
            contents=[
                types.Part.from_bytes(
                    data=img_b64,
                    mime_type='image/jpeg',
                ),
                'Voici une capture d’écran contenant une question de mathématiques ou d’ingénierie informatique. Donne uniquement la réponse selon les règles données.'
            ]
        )
        print(response.text)
    except Exception as e:
        print(f"❌ Erreur lors de l'appel à l'API : {e}")

    return response.text

# Fonction principale (appelée au raccourci)
def on_hotkey():
    print("\n🎬 Capture déclenchée !")
    img_b64 = capture_screen()
    save_image(img_b64)  # Optionnel : sauvegarde l'image
    result = obtenir_reponse(img_b64)


# Définir le raccourci global
print("🟢 Assistant overlay lancé. Appuie sur Ctrl + Alt + A pour analyser ton écran.")
keyboard.add_hotkey("ctrl+alt+a", on_hotkey)

keyboard.wait()  # bloque le script en fond, sans fenêtre
