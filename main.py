import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

app = FastAPI(title="Groq AI API")

@app.get("/")
def home():
    return {
        "status": True,
        "message": "API is running"
    }

@app.get("/api/ai/gpt")
async def gpt(q: str = Query(...)):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Reply in the same language as the user."
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
