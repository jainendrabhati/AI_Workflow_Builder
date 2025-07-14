
# Intelligent Workflow Builder

A No-Code/Low-Code web application that enables users to visually create and interact with intelligent AI workflows.

## Features

- Visual workflow builder using React Flow
- Support for 4 core components:
  - User Query Component
  - Knowledge Base Component (document processing)
  - LLM Engine Component (OpenAI GPT, Gemini)
  - Output Component (chat interface)
- Document upload and processing with embeddings
- Vector search using ChromaDB
- Real-time chat interface
- PostgreSQL database for persistence

## Tech Stack

- **Frontend**: React.js, React Flow, Tailwind CSS, Shadcn/ui
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy
- **AI Services**: OpenAI GPT/Embeddings, Google Gemini
- **Vector Store**: ChromaDB
- **Document Processing**: PyMuPDF
- **Web Search**: SerpAPI (optional)

## Required API Keys

You'll need the following API keys:

1. **OpenAI API Key** (Required)
   - For GPT models and text embeddings
   - Get it from: https://platform.openai.com/api-keys

2. **Google AI API Key** (Optional)
   - For Gemini models
   - Get it from: https://makersuite.google.com/app/apikey

3. **SerpAPI Key** (Optional)
   - For web search functionality
   - Get it from: https://serpapi.com/

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Quick Start with Docker

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd intelligent-workflow-builder
   ```

2. Create environment file:
   ```bash
   cp backend/.env.example .env
   ```

3. Add your API keys to the `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_AI_API_KEY=your_google_ai_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here
   ```

4. Start the application:
   ```bash
   docker-compose up -d
   ```

5. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development Setup

#### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Start PostgreSQL database (using Docker):
   ```bash
   docker run -d --name postgres-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=workflow_db -p 5432:5432 postgres:15
   ```

6. Run the backend:
   ```bash
   uvicorn app.main:app --reload
   ```

#### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

## Usage

1. **Landing Page**: Visit the homepage and click "Get Started"
2. **Create Workflow**: Create a new stack (workflow)
3. **Build Workflow**: Drag and drop components to build your workflow
4. **Configure Components**: Select components to configure their settings
5. **Upload Documents**: Upload PDFs to the Knowledge Base component
6. **Test Workflow**: Use the chat interface to test your workflow

## API Endpoints

### Workflows
- `GET /api/v1/workflows/` - Get all workflows
- `POST /api/v1/workflows/` - Create new workflow
- `PUT /api/v1/workflows/{id}` - Update workflow
- `GET /api/v1/workflows/{id}` - Get workflow by ID

### Documents
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents/search` - Search documents

### Chat
- `POST /api/v1/chat/` - Send chat message and execute workflow

## Architecture

The application follows a microservices architecture:

- **Frontend**: React SPA with React Flow for visual workflow building
- **Backend**: FastAPI with async processing
- **Database**: NeonPostgreSQL for data persistence
- **Vector Store**: ChromaDB for document embeddings
- **AI Services**: OpenAI and Google AI APIs


