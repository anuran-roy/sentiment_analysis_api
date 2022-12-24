from typing import Optional, List, Dict, Any, Tuple, Union
import random


async def infer(sentence: str) -> Any:  # Would make it async soon
    return random.choice(["positive", "negative"])  # Placeholder for the time being
