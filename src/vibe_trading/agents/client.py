import json
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

    def call_llm_with_tools(
        self,
        model_name: str,
        system_instruction: str,
        prompt: str,
        tools: list,
        tool_executor,
        max_iterations: int = 10,
    ) -> str:
        """Multi-turn agentic loop: LLM proposes tool calls, executor runs them, results fed back.

        Returns the final `assistant.content` string once the model stops emitting tool_calls.
        Raises RuntimeError if `max_iterations` is exhausted with the model still requesting tools.
        """
        model_str = get_litellm_model_string(self.provider, model_name)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        for iteration in range(max_iterations):
            logger.info(f"Tool-use loop iteration {iteration + 1}/{max_iterations} (model={model_str})")
            response = litellm.completion(
                model=model_str,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)

            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if not tool_calls:
                return assistant_msg.content

            for tool_call in tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    args_result = json.dumps({"error": f"Malformed tool arguments: {e}"})
                else:
                    logger.info(f"Executing tool: {tool_call.function.name}({args})")
                    args_result = tool_executor.execute(tool_call.function.name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": args_result,
                })

        raise RuntimeError(f"Agent exceeded max tool-call iterations ({max_iterations})")
