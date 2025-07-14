from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os
from dotenv import load_dotenv

from database import get_db, engine
from models import Base
from crud import StackCRUD, WorkflowCRUD
from schemas import StackCreate, StackResponse, WorkflowCreate, WorkflowResponse, ChatMessage, ChatResponse
from services.workflow_executor import WorkflowExecutor
from services.file_processor import FileProcessor
from services.knowledge_base_service import KnowledgeBaseService

load_dotenv()

import os
print(os.getenv("GEMINI_API_KEY"))

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="GenAI Stack Builder API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stack_crud = StackCRUD()
workflow_crud = WorkflowCRUD()
workflow_executor = WorkflowExecutor()
file_processor = FileProcessor()

@app.get("/")
async def root():
    return {"message": "GenAI Stack Builder API"}

@app.post("/stacks/", response_model=StackResponse)
async def create_stack(stack: StackCreate, db: Session = Depends(get_db)):
    """Create a new stack"""
    try:
        db_stack = stack_crud.create_stack(db, stack)
        return db_stack
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stacks/", response_model=List[StackResponse])
async def get_stacks(db: Session = Depends(get_db)):
    """Get all stacks"""
    return stack_crud.get_stacks(db)

@app.get("/stacks/{stack_id}", response_model=StackResponse)
async def get_stack(stack_id: str, db: Session = Depends(get_db)):
    """Get a specific stack"""
    stack = stack_crud.get_stack(db, stack_id)
    if not stack:
        raise HTTPException(status_code=404, detail="Stack not found")
    return stack

@app.put("/stacks/{stack_id}", response_model=StackResponse)
async def update_stack(stack_id: str, stack: StackCreate, db: Session = Depends(get_db)):
    """Update a stack"""
    updated_stack = stack_crud.update_stack(db, stack_id, stack)
    if not updated_stack:
        raise HTTPException(status_code=404, detail="Stack not found")
    return updated_stack

@app.delete("/stacks/{stack_id}")
async def delete_stack(stack_id: str, db: Session = Depends(get_db)):
    """Delete a stack"""
    if not stack_crud.delete_stack(db, stack_id):
        raise HTTPException(status_code=404, detail="Stack not found")
    return {"message": "Stack deleted successfully"}

@app.post("/workflows/", response_model=WorkflowResponse)
async def save_workflow(workflow: WorkflowCreate, db: Session = Depends(get_db)):
    """Save workflow with nodes and edges"""

    # try:
        # Validate workflow structure
    if not workflow.nodes:
        raise HTTPException(status_code=400, detail="Workflow must have at least one node")
    
    # Validate API keys and prompts
    for node in workflow.nodes:
        if node.data.componentType == "Knowledge Base":
            if not node.data.config.get("apiKey"):
                raise HTTPException(status_code=400, detail="API Key is required for Knowledge Base component")
        elif node.data.componentType == "LLM (OpenAI)":
            if not node.data.config.get("apiKey"):
                raise HTTPException(status_code=400, detail="API Key is required for LLM component")
            if not node.data.config.get("prompt"):
                raise HTTPException(status_code=400, detail="Prompt is required for LLM component")
        elif node.data.componentType == "User Query":
            if not node.data.config.get("placeholder"):
                raise HTTPException(status_code=400, detail="Query text is required for User Query component")
    
    # Validate workflow before saving
    validation_error = workflow_executor.validate_workflow(workflow.nodes, workflow.edges)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    db_workflow = workflow_crud.create_workflow(db, workflow)
    return db_workflow
        
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))


@app.post("/workflows/build")
async def build_workflow(workflow: WorkflowCreate, db: Session = Depends(get_db)):
    """Build and execute a workflow"""
    # try:
    # Validate workflow structure
    if not workflow.nodes:
        raise HTTPException(status_code=400, detail="Workflow must have at least one node")
    print(1)
    # Validate API keys and prompts
    for node in workflow.nodes:
        if node.data.componentType == "Knowledge Base":
            if not node.data.config.get("apiKey"):
                raise HTTPException(status_code=400, detail="API Key is required for Knowledge Base component")
        elif node.data.componentType == "LLM (OpenAI)":
            if not node.data.config.get("apiKey"):
                raise HTTPException(status_code=400, detail="API Key is required for LLM component")
            if not node.data.config.get("prompt"):
                raise HTTPException(status_code=400, detail="Prompt is required for LLM component")
        elif node.data.componentType == "User Query":
            if not node.data.config.get("placeholder"):
                raise HTTPException(status_code=400, detail="Query text is required for User Query component")
    
    # Validate workflow before building
    print(2)
    validation_error = workflow_executor.validate_workflow(workflow.nodes, workflow.edges)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    print(3)
    # Create/update workflow in database
    db_workflow = workflow_crud.create_workflow(db, workflow)
    print(4)
    # Find user query from nodes
    user_query = None
    for node in workflow.nodes:
        if node.data.componentType == "User Query":
            user_query = node.data.config.get("placeholder", "")
            break
    print(5)
    if not user_query:
        raise HTTPException(status_code=400, detail="User Query component is required")
    
    # Execute the workflow with stack_id context
    print(6)
    result = await workflow_executor.execute_workflow(workflow.nodes, workflow.edges, user_query, workflow.stack_id)
    
    return {
        "message": "Workflow built and executed successfully",
        "workflow_id": db_workflow.id,
        "stack_id": db_workflow.stack_id,
        "result": result
    }
        
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{stack_id}", response_model=Optional[WorkflowResponse])
async def get_workflow(stack_id: str, db: Session = Depends(get_db)):
    """Get workflow for a stack"""
    return workflow_crud.get_workflow_by_stack(db, stack_id)

@app.post("/workflows/{stack_id}/execute")
async def execute_workflow(stack_id: str, message: ChatMessage, db: Session = Depends(get_db)):
    """Execute workflow with user query"""
    # try:
    workflow = workflow_crud.get_workflow_by_stack(db, stack_id)
    print(1)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found for this stack")
    
    # Execute the workflow
    response = await workflow_executor.execute_workflow_chat(
        workflow.nodes, 
        workflow.edges, 
        message.message
    )
    print(22222)
    return ChatResponse(response=response)
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.post("/upload-file/{node_id}")
async def upload_file(
    node_id: str, 
    file: UploadFile = File(...),
    api_key: str = Form(...),
    embedding_model: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload and process file for knowledge base"""

    # try:
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Process the file
    chunks = await file_processor.process_pdf(file)
    print("____________________________________________________________________________________+++")
    print(len(chunks))
    print("____________________________________________________________________________________+++")
    
    # Store embeddings with API key and embedding model
    knowledge_base_service = KnowledgeBaseService()
    
    await knowledge_base_service.store_embeddings(node_id, chunks, api_key, embedding_model)
    
    return {"message": f"File {file.filename} processed successfully", "chunks_count": len(chunks)}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)