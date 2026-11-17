from openai import OpenAI
import os
import json
import time
from dotenv import load_dotenv
import prompts

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
)

text = input("Введите текст статьи для категоризации:\n")


def log(data):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "calls.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    content = {"records": []}
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            with open(log_path, "r", encoding="utf-8") as file:
                content = json.load(file)
        except json.JSONDecodeError:
            # файл битый — начинаем лог заново, старое сохраняем рядом
            os.replace(log_path, log_path + ".broken")

    content.setdefault("records", []).append(data)
    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


response = client.responses.create(
    model = prompts.MODEL,
    instructions = prompts.PROMPT_INSTRUCTION,
    input = text,
    temperature = prompts.TEMPERATURE,
    text={
        "format": {
            "type": "json_schema",
            "name": "mythology_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mythology": {"type": "string", "enum": list(prompts.MYTHOLOGIES)},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["mythology", "confidence"],
                "additionalProperties": False,
            },
        }
    },
)

data = {
    "timestamp": time.time(),
    "status": "OK",
    "model": prompts.MODEL,
    "temperature": prompts.TEMPERATURE,
    "instruction": prompts.PROMPT_INSTRUCTION,
    "prompt": text,
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
    "total_tokens": response.usage.total_tokens,
    "output": response.output_text,
    "taxonomy_id": 0,
}

output = json.loads(response.output_text)

if (output["mythology"] in prompts.MYTHOLOGIES):
    data["taxonomy_id"] = prompts.MYTHOLOGIES[output["mythology"]]
    if (output["confidence"] == prompts.MIN_LEVEL):
        data["status"] = "REVIEW"
else:
    data['status'] = "ERROR"

log(data)