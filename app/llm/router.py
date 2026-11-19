
from litellm import Router

from app.config import GROQ_API_KEY, OPENAI_API_KEY


model_list = [
    {
        "model_name": "travel-llm",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            "api_key": GROQ_API_KEY,
        },
        "model_info": {
            "id": "groq-gpt-oss-20b",
        },
    },
    {
        "model_name": "travel-llm",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-120b",
            "api_key": GROQ_API_KEY,
        },
        "model_info": {
            "id": "groq-gpt-oss-120b",
        },
    },
]

# Optional OpenAI fallback.
# if OPENAI_API_KEY:
#     model_list.append({
#         "model_name": "travel-llm",
#         "litellm_params": {
#             "model": "openai/gpt-4o-mini",
#             "api_key": OPENAI_API_KEY,
#         },
#         "model_info": {
#             "id": "openai-gpt4o-mini",
#         },
#     })


router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",
    num_retries=2,
    timeout=60,
)
