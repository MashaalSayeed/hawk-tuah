from pydantic import BaseModel
from typing import Dict, Any


class CommandRequest(BaseModel):
    command: str
    arguments: Dict[str, Any]


class ChangeStatusRequest(BaseModel):
    status: str