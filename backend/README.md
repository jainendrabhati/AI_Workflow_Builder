# GenAI Stack Builder Backend

FastAPI backend for the GenAI Stack Builder application with PostgreSQL database and workflow execution engine.

## Features

- **REST API** for stack and workflow management
- **PostgreSQL** database with SQLAlchemy ORM
- **Workflow Execution Engine** with validation
- **File Processing** for PDF documents with PyMuPDF
- **Vector Store** integration with ChromaDB
- **LLM Integration** with OpenAI and Gemini
- **Web Search** integration with SerpAPI
- **Docker** containerization support

## Setup Instructions

### Option 1: Local Development

1. **Install Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Clone and Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Database Setup (Neon PostgreSQL)**
   - Create a Neon PostgreSQL database at https://neon.tech
   - Copy your connection string

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file:
   ```env
   DATABASE_URL=postgresql://username:password@host:5432/database_name
   OPENAI_API_KEY=your_openai_api_key_here
   SERPAPI_KEY=your_serpapi_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run Database Migrations**
   ```bash
   # Tables will be created automatically when you start the app
   # Or use Alembic for more control:
   alembic upgrade head
   ```

6. **Start the Server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Option 2: Docker Development

1. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (DATABASE_URL will be set automatically)
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **View Logs**
   ```bash
   docker-compose logs -f backend
   ```

### Option 3: Production Deployment

1. **Build Docker Image**
   ```bash
   docker build -t genai-stack-backend .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e DATABASE_URL="your_neon_postgres_url" \
     -e OPENAI_API_KEY="your_openai_key" \
     -e SERPAPI_KEY="your_serpapi_key" \
     -e GEMINI_API_KEY="your_gemini_key" \
     genai-stack-backend
   ```

## API Endpoints

### Stacks
- `POST /stacks/` - Create a new stack
- `GET /stacks/` - Get all stacks
- `GET /stacks/{stack_id}` - Get specific stack
- `PUT /stacks/{stack_id}` - Update stack
- `DELETE /stacks/{stack_id}` - Delete stack

### Workflows
- `POST /workflows/` - Save workflow (nodes and edges)
- `GET /workflows/{stack_id}` - Get workflow for stack
- `POST /workflows/{stack_id}/execute` - Execute workflow with query

### File Upload
- `POST /upload-file/{stack_id}` - Upload PDF file for knowledge base

### Health Check
- `GET /health` - Health check endpoint

## API Usage Examples

### Create a Stack
```bash
curl -X POST "http://localhost:8000/stacks/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Customer Support AI",
    "description": "AI assistant for customer support queries"
  }'
```

### Save Workflow
```bash
curl -X POST "http://localhost:8000/workflows/" \
  -H "Content-Type: application/json" \
  -d '{
    "stack_id": "your-stack-id",
    "nodes": [...],
    "edges": [...]
  }'
```

### Execute Workflow
```bash
curl -X POST "http://localhost:8000/workflows/your-stack-id/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I reset my password?"
  }'
```

## Database Schema

### Tables
- **stacks** - Store stack information
- **workflows** - Store workflow definitions (nodes and edges)
- **documents** - Store uploaded document metadata
- **chat_logs** - Store chat history

## Workflow Validation

The backend validates workflows before execution:

1. **Required Components**: Must have User Query and Output components
2. **Configuration Validation**: 
   - Knowledge Base: Requires API key and embedding model
   - LLM: Requires API key, model, and prompt
3. **Connection Validation**: Must have valid path from User Query to Output

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- `400` - Validation errors, missing configuration
- `404` - Resource not found
- `500` - Server errors, workflow execution failures

## Development

### Adding New Components

1. Update `workflow_executor.py` validation logic
2. Add execution logic in `_execute_node` method
3. Update schemas if needed

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## Testing

Access the API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure your Neon PostgreSQL URL is correct
2. **API Keys**: Verify all required API keys are set in `.env`
3. **Ports**: Make sure port 8000 is available
4. **Dependencies**: Run `pip install -r requirements.txt` if imports fail

### Logs

```bash
# View application logs
docker-compose logs -f backend

# Check database connection
docker-compose exec backend python -c "from database import engine; print(engine.execute('SELECT 1').scalar())"
```