from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional, Literal


class InferenceResponse(BaseModel):
    """Inference response schema."""

    sentence: str
    sentiment: Literal["positive", "negative", "neutral"]
