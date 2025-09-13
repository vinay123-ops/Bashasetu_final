import requests

url = "http://127.0.0.1:8000/translate"  # Use 127.0.0.1 if testing locally
# If testing from another device, use the server's IP, e.g., "http://192.168.1.10:8000/translate"

payload = {
  "sentences": [
    "Chief of Defence Staff General Anil Chauhan visited the Defence Services Staff College, Wellington in Tamil Nadu on July 19, 2025.",
    "He addressed the student officers of 81st Staff Course, Permanent staff of the College and station officers of Wellington.",
    "The CDS delivered a talk on Operation Sindoor and emphasised on important aspects of Tri-Services synergy demonstrated during the successful operations by the Indian Armed Forces.",
    "Later, while interacting with the faculty of the college, General Anil Chauhan laid stress on Integration & Jointness imperatives, Capability Development, Aatmanirbharta and an in-depth understanding of the transformative changes being pursued in the military.",
    "The CDS was also briefed by the DSSC Commandant Lt Gen Virendra Vats on the ongoing training activities at the College, where emphasis is being laid on fostering jointness & inter-services awareness, specifically with the institutionalisation of the Deep Purple Division.",
    "The 45-week 81st Staff Course is presently underway at the College.",
    "The present course comprises 500 student officers, including 45 from 35 friendly countries."
  ],
  "src_lang": "eng_Latn",
  "tgt_lang": "hin_Deva"
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
