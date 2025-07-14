from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Stack(Base):
    __tablename__ = "stacks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with workflows
    workflows = relationship("Workflow", back_populates="stack", cascade="all, delete-orphan")

class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stack_id = Column(String, ForeignKey("stacks.id"), nullable=False)
    nodes = Column(JSON, nullable=False)  # Store React Flow nodes
    edges = Column(JSON, nullable=False)  # Store React Flow edges
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with stack
    stack = relationship("Stack", back_populates="workflows")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stack_id = Column(String, ForeignKey("stacks.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatLog(Base):
    __tablename__ = "chat_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    stack_id = Column(String, ForeignKey("stacks.id"), nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    
