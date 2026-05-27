import os
import logging
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

litellm.telemetry = False

def get_litellm_model_string(provider: str, model: str) -> str:
    """Converts provider and model parameters to standard LiteLLM model identifiers."""
    provider = provider.lower()
    if provider == "gemini":
        return f"gemini/{model}"
    elif provider == "openai":
        return f"openai/{model}"
    elif provider == "anthropic":
        return f"anthropic/{model}"
    elif provider == "ollama":
        return f"ollama/{model}"
    return model

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        
        # Dynamic key validation for active provider only
        if self.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please check your .env file.")
        elif self.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        elif self.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please check your .env file.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM request failed. Retrying in {retry_state.next_action.sleep} seconds... (Attempt {retry_state.attempt_number})"
        )
    )
    def call_llm(
        self,
        model_name: str,
        system_instruction: str,
        prompt: str,
        response_schema: type = None
    ) -> str:
        """
        Invokes the configured LLM provider via LiteLLM and returns the raw JSON string content.
        """
        model_str = get_litellm_model_string(self.provider, model_name)
        logger.info(f"Calling LLM provider={self.provider} model={model_str}...")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": model_str,
            "messages": messages,
            "temperature": 0.1,
        }
        
        if response_schema:
            kwargs["response_format"] = response_schema
            
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
