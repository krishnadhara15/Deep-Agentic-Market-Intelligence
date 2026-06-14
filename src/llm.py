"""LLM factory supporting OpenAI, Gemini, and Tavily-only modes."""

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Config


def get_llm(config: Config, for_report: bool = False) -> BaseChatModel:
    """Return the configured chat model for the active LLM provider."""
    if config.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = config.gemini_model_report if for_report else config.gemini_model
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=config.gemini_api_key,
        )

    from langchain_openai import ChatOpenAI

    model = config.openai_model_report if for_report else config.openai_model
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=config.openai_api_key,
    )
