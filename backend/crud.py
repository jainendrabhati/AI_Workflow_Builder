from sqlalchemy.orm import Session
from sqlalchemy import desc
from models import Stack, Workflow, Document, ChatLog
from schemas import StackCreate, WorkflowCreate
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

class StackCRUD:
    def create_stack(self, db: Session, stack: StackCreate) -> Stack:
        db_stack = Stack(name=stack.name, description=stack.description)
        db.add(db_stack)
        db.commit()
        db.refresh(db_stack)
        return db_stack
    
    def get_stacks(self, db: Session) -> List[Stack]:
        return db.query(Stack).order_by(desc(Stack.updated_at)).all()
    
    def get_stack(self, db: Session, stack_id: str) -> Optional[Stack]:
        return db.query(Stack).filter(Stack.id == stack_id).first()
    
    def update_stack(self, db: Session, stack_id: str, stack: StackCreate) -> Optional[Stack]:
        db_stack = self.get_stack(db, stack_id)
        if db_stack:
            db_stack.name = stack.name
            db_stack.description = stack.description
            db.commit()
            db.refresh(db_stack)
        return db_stack
    
    def delete_stack(self, db: Session, stack_id: str) -> bool:
        db_stack = self.get_stack(db, stack_id)
        if db_stack:
            db.delete(db_stack)
            db.commit()
            return True
        return False

class WorkflowCRUD:

    def create_workflow(self, db: Session, workflow: WorkflowCreate) -> Workflow:
        # Convert Pydantic models to dict for JSON storage
        nodes_dict = [node.dict() for node in workflow.nodes]
        edges_dict = [edge.dict() for edge in workflow.edges]

        try:
            # ✅ Check if stack exists
            stack = db.query(Stack).filter(Stack.id == workflow.stack_id).first()

            if not stack:
                # ✅ Create a new stack with the provided stack_id
                stack = Stack(
                    id=workflow.stack_id,  # Use provided stack_id
                    name="Untitled Stack",
                    description="Auto-created during workflow creation",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(stack)
                db.flush()  # Ensures the stack is available for the foreign key

            # Check if workflow already exists for this stack
            existing_workflow = db.query(Workflow).filter(Workflow.stack_id == workflow.stack_id).first()

            if existing_workflow:
                # Update existing workflow
                existing_workflow.nodes = nodes_dict
                existing_workflow.edges = edges_dict
                existing_workflow.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing_workflow)
                return existing_workflow
            else:
                # Create new workflow
                db_workflow = Workflow(
                    stack_id=workflow.stack_id,
                    nodes=nodes_dict,
                    edges=edges_dict,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(db_workflow)
                db.commit()
                db.refresh(db_workflow)
                return db_workflow

        except SQLAlchemyError as e:
            db.rollback()
            print(f"[create_workflow] Database error: {e}")
            raise HTTPException(
                status_code=500,
                detail="An error occurred while creating or updating the workflow."
            )

    def get_workflow_by_stack(self, db: Session, stack_id: str) -> Optional[Workflow]:
        return db.query(Workflow).filter(Workflow.stack_id == stack_id).first()
    
    def get_workflow(self, db: Session, workflow_id: str) -> Optional[Workflow]:
        return db.query(Workflow).filter(Workflow.id == workflow_id).first()

class DocumentCRUD:
    def create_document(self, db: Session, stack_id: str, filename: str, file_path: str, chunks_count: int) -> Document:
        db_document = Document(
            stack_id=stack_id,
            filename=filename,
            file_path=file_path,
            chunks_count=chunks_count
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        return db_document
    
    def get_documents_by_stack(self, db: Session, stack_id: str) -> List[Document]:
        return db.query(Document).filter(Document.stack_id == stack_id).all()

class ChatLogCRUD:
    def create_chat_log(self, db: Session, stack_id: str, user_message: str, ai_response: str) -> ChatLog:
        db_chat_log = ChatLog(
            stack_id=stack_id,
            user_message=user_message,
            ai_response=ai_response
        )
        db.add(db_chat_log)
        db.commit()
        db.refresh(db_chat_log)
        return db_chat_log
    
    def get_chat_history(self, db: Session, stack_id: str) -> List[ChatLog]:
        return db.query(ChatLog).filter(ChatLog.stack_id == stack_id).order_by(ChatLog.created_at).all()