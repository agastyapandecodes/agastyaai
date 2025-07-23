import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS for all origins. In a production environment, you should restrict this
# to specific frontend origins for better security (e.g., CORS(app, resources={r"/chat/*": {"origins": "https://your-frontend-domain.netlify.app"}})).
CORS(app)

# --- Configuration ---
# IMPORTANT: This line now expects the Gemini API key to be set as an environment variable
# on your hosting platform (e.g., Render).
# DO NOT hardcode your API key here for public deployment.
# Example for Render: In your Render service settings, add an environment variable named GEMINI_API_KEY
# with your actual API key as its value.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_gemini_response(prompt: str) -> str:
    """
    Sends a prompt to the Gemini API and returns the generated text response.

    Args:
        prompt (str): The user's input message.

    Returns:
        str: The AI's generated response, or an error message if something goes wrong.
    """
    if not GEMINI_API_KEY:
        print("Error: Gemini API Key environment variable is not set on the server.")
        return "Error: Backend configuration issue. Gemini API Key is missing."

    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "key": GEMINI_API_KEY
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        # Make the POST request to the Gemini API
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse the JSON response
        result = response.json()

        # Extract the generated text
        if result and "candidates" in result and len(result["candidates"]) > 0 and \
           "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"] and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print(f"DEBUG: Unexpected API response structure: {json.dumps(result, indent=2)}")
            return "Sorry, I couldn't get a valid response from the AI. Please try again."

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return f"An error occurred while connecting to the AI service: {e}"
    except json.JSONDecodeError:
        print("Error parsing Gemini API response.")
        return "An error occurred while processing the AI's response."
    except Exception as e:
        print(f"An unexpected error occurred in get_gemini_response: {e}")
        return f"An unexpected error occurred: {e}"

# --- API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint to receive user messages and return AI responses.
    Expects a JSON payload with a 'message' field.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    print(f"Received message from frontend: {user_message}")
    ai_response = get_gemini_response(user_message)
    print(f"Sending response to frontend: {ai_response}")

    return jsonify({"response": ai_response})

# --- Run the Flask App ---
if __name__ == "__main__":
    # When deployed, the hosting service (e.g., Render) will set the PORT environment variable.
    # Locally, it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
