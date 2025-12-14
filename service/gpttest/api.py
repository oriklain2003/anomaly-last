from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """
You are ChatGPT, the same assistant that appears on chat.openai.com.
Behave like the original ChatGPT:
- friendly but professional
- clear, structured answers
- think step-by-step internally (do NOT reveal chain-of-thought)
- format answers with markdown when useful
- if user uploads an image, describe what you see and help them with it
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(message: str = Form(...), image: UploadFile = None):
    content = [{"type": "text", "text": message}]
    print("started")
    if image:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.content_type};base64,{img_b64}"
            }
        })
    print("wait")

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    )
    print("do")

    return {"answer": response.choices[0].message.content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=789)
