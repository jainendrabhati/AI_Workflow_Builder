from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class StackCreate(BaseModel):
    name: str
    description: str

class StackResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class NodeData(BaseModel):
    label: str
    componentType: str
    config: Dict[str, Any] = {}

class FlowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: NodeData

class FlowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Optional[str] = None

class WorkflowCreate(BaseModel):
    stack_id: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]

class WorkflowResponse(BaseModel):
    id: str
    stack_id: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class FileUploadResponse(BaseModel):
    message: str
    chunks_count: int