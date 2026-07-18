"""
Gemini API key rotation — automatically switches to the next key on 429 errors.

Add multiple keys to .env as:
  GEMINI_API_KEY=key1
  GEMINI_API_KEY_2=key2
  GEMINI_API_KEY_3=key3
  ...

Import this module BEFORE any litellm calls to activate rotation.
"""
import os
import litellm
from litellm import completion as _original_completion

_key_index = 0


def _get_gemini_keys() -> list[str]:
    keys = []
    base = os.getenv("GEMINI_API_KEY", "")
    if base:
        keys.append(base)
    i = 2
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if k:
            keys.append(k)
            i += 1
        else:
            break
    return keys


def _rotating_completion(*args, **kwargs):
    global _key_index

    model = kwargs.get("model", args[0] if args else "")
    provider = kwargs.get("custom_llm_provider", "")
    is_gemini = "gemini" in str(model).lower() or "gemini" in str(provider).lower()

    if not is_gemini:
        return _original_completion(*args, **kwargs)

    keys = _get_gemini_keys()
    if not keys:
        return _original_completion(*args, **kwargs)

    last_error = None
    for _ in range(len(keys)):
        current_key = keys[_key_index % len(keys)]
        os.environ["GEMINI_API_KEY"] = current_key
        try:
            return _original_completion(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "RateLimitError" in type(e).__name__ or "RESOURCE_EXHAUSTED" in err:
                _key_index += 1
                last_error = e
                print(f"[key_rotation] Key {_key_index % len(keys) + 1}/{len(keys)} hit rate limit, rotating to next key.")
                continue
            raise

    raise last_error


litellm.completion = _rotating_completion
print(f"[key_rotation] Gemini key rotation active — {len(_get_gemini_keys())} key(s) loaded.")
