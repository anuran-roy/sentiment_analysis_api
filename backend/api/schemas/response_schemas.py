from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional, Literal


class InferenceResponse(BaseModel):
    sentence: str
    sentiment: Literal["positive", "negative", "neutral"]
