from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional, Literal


class InferenceRequest(BaseModel):
    """Inference request schema."""

    sentence: str
