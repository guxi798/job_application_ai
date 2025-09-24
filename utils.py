from crewai import LLM
from litellm import completion
import logging

PRICE_PER_1M_CACHE_HIT_INPUT = 0.5  # RMB
PRICE_PER_1M_CACHE_MISS_INPUT = 4
PRICE_PER_1M_OUTPUT = 12

class LoggedLLM(LLM):
    def __init__(self, model: str, base_url: str, api_key: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.usage_totals = {
            "cost": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def call(self, messages: str, **kwargs) -> str:
        # Prepare the message for LiteLLM
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Call LiteLLM's completion function
        response = completion(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            messages=messages
        )

        # Extract the response content and token usage
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0)
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        cost = (cache_hit_tokens / 1_000_000 * PRICE_PER_1M_CACHE_HIT_INPUT) +  \
               (cache_miss_tokens / 1_000_000 * PRICE_PER_1M_CACHE_MISS_INPUT) + \
               (completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT)

        self.usage_totals["cost"] += cost
        self.usage_totals["prompt_cache_hit_tokens"] += cache_hit_tokens
        self.usage_totals["prompt_cache_miss_tokens"] += cache_miss_tokens
        self.usage_totals["completion_tokens"] += completion_tokens
        self.usage_totals["total_tokens"] += total_tokens
        
        # Log the token usage
        logging.info(f"[USAGE] est. cost={cost:.6f}, cache_hit={cache_hit_tokens}, cache_miss={cache_miss_tokens}, completion={completion_tokens}, total={total_tokens}")

        return content

# Configure LiteLLM with DeepSeek-R1 model
# deepseek_llm = LLM(
#     model="deepseek/deepseek-chat",
#     base_url="https://api.deepseek.com/v1",
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )