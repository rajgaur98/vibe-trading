import os
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from langfuse import get_client

logger = logging.getLogger(__name__)

# Initialize OpenTelemetry instrumentation for Google GenAI
try:
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    GoogleGenAIInstrumentor().instrument()
    # Trigger Langfuse client initialization to set up the OTel span processor
    _ = get_client()
except Exception as e:
    logger.warning(f"Failed to initialize Google GenAI OpenTelemetry instrumentor: {e}")

class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")
        
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),  # Catch-all for API errors/rate limits
        before_sleep=lambda retry_state: logger.warning(
            f"Gemini API request failed. Retrying in {retry_state.next_action.sleep} seconds... (Attempt {retry_state.attempt_number})"
        )
    )
    def call_gemini(
        self,
        model_name: str,
        system_instruction: str,
        prompt: str,
        response_schema: type = None
    ) -> str:
        """
        Invokes Gemini with system instruction, prompt, and optional Pydantic response schema.
        Includes automatic retry logic using tenacity.
        """
        logger.info(f"Calling Gemini model {model_name}...")
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,  # Low temperature for analytical consistency
        )
        
        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        
        return response.text
