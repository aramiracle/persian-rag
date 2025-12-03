from typing import List, Optional
import threading

from loguru import logger
from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.core.config import settings
from backend.schemas.schemas import SearchResult


class LLMService:
    _instance: Optional["LLMService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LLMService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            timeout=getattr(settings.llm, 'timeout', 60.0),
        )
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.max_completion_tokens = settings.llm.max_completion_tokens
        self.context_window = settings.llm.context_window
        
        # Token estimation constants (Persian text)
        # Persian/Arabic script: ~0.5-0.7 tokens per character
        # Mixed Persian/English: ~0.6 tokens per character (conservative)
        self.chars_per_token = 2.0
        
        self._initialized = True

    async def close(self) -> None:
        await self._client.close()

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    async def _call(self, messages: List[dict], max_tokens: Optional[int] = None) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=max_tokens or self.max_completion_tokens,
        )
        return response.choices[0].message.content or ""

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count based on character count.
        Conservative approximation for Persian/mixed content.
        """
        return int(len(text) / self.chars_per_token)

    def _truncate_context(self, system_prompt: str, question: str, documents: List[SearchResult]) -> str:
        """
        Dynamically builds context to fit within the context window.
        """
        sys_tokens = self._estimate_tokens(system_prompt)
        # Approximate tokens for question and answer structure
        q_tokens = self._estimate_tokens(f"Question: {question}\n\nAnswer:")
        safety_margin = 200
        
        available_tokens = self.context_window - (sys_tokens + q_tokens + self.max_completion_tokens + safety_margin)
        
        if available_tokens <= 0:
            logger.warning("Context window too small for prompt.")
            return "Context ignored due to token limits."

        used_tokens = 0
        final_parts = []
        
        for i, r in enumerate(documents, 1):
            d = r.document if not isinstance(r.document, dict) else r.document
            agency = d.get("news_agency_name", "") if isinstance(d, dict) else getattr(d, "news_agency_name", "")
            title = d.get("title", "") if isinstance(d, dict) else getattr(d, "title", "")
            content = d.get("content", "") if isinstance(d, dict) else getattr(d, "content", "")
            
            entry = f"[{i}] {agency} - {title}\n{content}"
            entry_tokens = self._estimate_tokens(entry)
            
            # Overhead for separator
            overhead = 5 if final_parts else 0 
            
            if used_tokens + entry_tokens + overhead > available_tokens:
                # Truncate final document if it fits partially
                remaining = available_tokens - used_tokens - overhead
                if remaining > 50:  # Only if meaningful amount left
                    # Convert tokens back to approximate characters
                    chars = int(remaining * self.chars_per_token)
                    entry = entry[:chars] + "... (truncated)"
                    final_parts.append(entry)
                break
            
            final_parts.append(entry)
            used_tokens += entry_tokens + overhead

        if not final_parts:
            return "No relevant documents."

        return "\n---\n".join(final_parts)

    async def generate_answer(self, question: str, context_documents: List[SearchResult]) -> str:
        try:
            sys_prompt = self._system_prompt()
            
            context = self._truncate_context(sys_prompt, question, context_documents)
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ]
            return await self._call(messages)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _system_prompt(self) -> str:
        return (
            "You are a QA assistant answering questions ONLY based on the provided Persian news documents.\n"
            "\nRULES:\n"
            "1. Use ONLY the text inside the provided documents. Do NOT use prior knowledge.\n"
            "2. Every factual statement MUST cite a document number like: [1], [2]\n"
            "3. If the answer is NOT found, reply exactly:\n"
            "«اطلاعات مربوطه در اسناد موجود نیست.»\n"
            "4. Respond fully in Persian.\n"
            "5. Keep answers short and direct.\n"
            "6. Think briefly. Do NOT perform long reasoning.\n"
            "7. NEVER reveal your reasoning, chain of thought, or internal steps.\n"
            "\nOUTPUT FORMAT:\n"
            "Final answer in Persian with citations, e.g.:\n"
            "«متن پاسخ ... [1][3]»\n"
        )


_llm_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service