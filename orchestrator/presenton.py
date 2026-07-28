import os
import requests

# ==============================
# Replace with your Presenton API token
# ==============================
API_TOKEN = ""

# Presenton API endpoint
url = "https://api.presenton.ai/api/v3/presentation/generate"

# Headers
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "content": "Create a professional 6-slide presentation on Artificial Intelligence in Healthcare.",
    "n_slides": 6,
    "language": "English",
    "standard_template": "dynamic"
    
}

# Send request
response = requests.post(url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)

# If presentation was generated successfully
if response.status_code == 200:

    response_json = response.json()

    # Get download URL
    download_url = response_json["path"]

    # Download the PowerPoint
    ppt = requests.get(download_url)

    # Create outputs folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Save presentation
    output_file = os.path.join("outputs", "presentation.pptx")

    with open(output_file, "wb") as f:
        f.write(ppt.content)

    print(f"\nPresentation downloaded successfully!")
    print(f"Saved at: {output_file}")

    # Optional: print useful info
    print("\nPresentation ID:", response_json["presentation_id"])
    print("Edit URL:", response_json["edit_path"])
    print("Credits Consumed:", response_json["credits_consumed"])

else:
    print("\nPresentation generation failed.")