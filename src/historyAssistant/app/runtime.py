from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Tuple

import openai

from historyAssistant.config import load_all_configs
from historyAssistant.llm import build_openai_client
from historyAssistant.llm.llm_client import get_default_model_name

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

MODEL = get_default_model_name()


@lru_cache(maxsize=1)
def load_runtime() -> Tuple[Any, Any]:
    bundle = load_all_configs()
    client = build_openai_client()
    return bundle, client
