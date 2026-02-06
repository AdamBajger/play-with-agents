import httpx
from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# query which models are available at the specified endpoint
APU_URI = "https://llm.ai.e-infra.cz/v1/"
token = Path('api_key').read_text().strip()
model_id = 'deepseek-v3.2-thinking'
# do this with httpx
# curl -H "Authorization: Bearer ${E_INFRA_API_TOKEN}" https://llm.ai.e-infra.cz/v1/models | jq .data[].id
models = httpx.get(
    f"{APU_URI}models",
    headers={"Authorization": f"Bearer {token}"},
).json()
available_models = models['data']


if __name__ == "__main__":
    print("Available models at the specified endpoint:")
    for m in available_models:
        print(f"- {m['id']}")
