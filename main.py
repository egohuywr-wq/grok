import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI(
    title="Groq AI API",
    version="1.0"
)

@app.get("/")
async def home():
    return {
        "status": True,
        "message": "Groq AI API Running"
    }

@app.get("/api/ai/gpt")
async def chat(q: str = Query(...)):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Reply in the same language as the user."
                },
                {
                    "role": "user",
                    "content": q
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        return {
            "status": True,
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
