from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional, Literal


class InferenceRequest(BaseModel):
    sentence: str
