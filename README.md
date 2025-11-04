# HRM Agent 2.0
An intelligent AI-powered Human Resource Management assistant that provides personalized responses to employee queries. Built with FastAPI and Mistral AI, the system retrieves real-time data from the database based on the logged-in user's credentials and generates contextual, personalized answers. All HR features are accessible through a conversational interface, putting comprehensive HR support right at employees' fingertips.

First, an employee logs in. After logging in, they can access a chat interface that displays the full history of their conversations, similar to ChatGPT. When the user asks a query, it is sent to the API, which processes the request and determines the correct database collection to retrieve relevant data. The system fetches the data from the collection, and this data, along with the user’s query, is processed by Mistral AI. The AI then generates a personalized response based on the logged-in user’s context, which is sent back to the user as an answer.
## 🌟 Key Features

- **🤖 AI Chat Agent**: Graph-based conversational AI using LangGraph and Mistral AI
- **🔐 OTP Authentication**: Secure JWT-based authentication with email OTP verification
- **🔍 Vector Search**: Pinecone integration for semantic document search
- **📊 MongoDB Storage**: User data and chat history management
- **🎨 Web Interface**: Modern responsive UI with Jinja2 templates
- **📧 Email Service**: Automated OTP delivery via SMTP
- **📝 Logging**: Comprehensive logging for monitoring and debugging
- **🚀 Production Ready**: CORS enabled, async operations, middleware configured

## 📋 Prerequisites

- Python 3.12 or higher
- MongoDB (local or MongoDB Atlas)
- Pinecone account and API key
- Mistral AI API key
- SMTP server access (for OTP emails)

## 🏗️ Project Structure

```
hrm-agent-2-0/
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── setup.sh                    # Automated setup script
├── run.sh                      # Server startup script
├── .env                        # Environment variables (create this)
│
├── config/                     # Configuration modules
│   ├── database.py            # MongoDB connection
│   ├── pinecone.py            # Pinecone vector DB setup
│   └── logger.py              # Logging configuration
│
├── core/                      # Core business logic
│   ├── auth.py                # JWT authentication
│   ├── graph_agent.py         # LangGraph agent implementation
│   ├── middleware.py          # FastAPI middleware
│   ├── otp_service.py         # OTP generation and verification
│   └── __init__.py
│
├── models/                    # Pydantic data models
│   └── index.py               # Request/response schemas
│
├── routes/                    # API endpoints
│   ├── auth.py                # Authentication routes
│   ├── chat.py                # Chat API routes
│   ├── mistralai.py           # Mistral AI routes
│   └── web.py                 # Web page routes
│
├── services/                  # External services
│   ├── get_user_data.py       # User data retrieval
│   ├── mistralai.py           # Mistral AI service
│   └── otp_service.py         # OTP email service
│
├── templates/                 # HTML templates
│   ├── base.html              # Base layout
│   ├── home.html              # Landing page
│   ├── login.html             # Login/signup page
│   └── chat.html              # Chat interface
│
├── scripts/                   # Utility scripts
│   └── collections_vector.py  # Vector management
│
├── utils/                     # Helper utilities
│   └── index.py               # Common functions
│
├── uploads/                   # File uploads
│   ├── input/                 # Uploaded files
│   └── output/                # Processed files
│
└── logs/                      # Application logs
    └── hrm_agent.log          # Main log file
```



## System Flow Diagram

```text
┌────────────────────────────┐
│        Employee User        │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│        Login System         │
│ (Auth via user credentials) │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│     Chat Interface (UI)     │
│ - Shows chat history        │
│ - Accepts new queries       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│       FastAPI Backend       │
│ - Receives user query       │
│ - Identifies logged-in user │
│ - Sends query for processing│
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│          Processing & Intelligence Layer    │
│--------------------------------------------│
│ 1. Determine correct database collection    │
│ 2. Retrieve relevant HR data (MongoDB)      │
│ 3. Combine user query + HR data             │
│ 4. Send contextual prompt to Mistral AI     │
│ 5. Receive personalized AI-generated answer │
└────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────┐
│     Response Generation     │
│ - Formats personalized reply│
│ - Stores conversation in DB │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│    Chat Interface (UI)      │
│  - Displays AI’s response   │
│  - Updates chat history     │
└────────────────────────────┘


## How It Works

1. **Employee Login:**  
   - The user logs into the system, which validates their credentials.

2. **Access Chat Interface:**  
   - The chat interface loads the user's conversation history.  
   - The employee can ask HR-related questions directly in the chat.

3. **Query Processing:**  
   - The query is sent to the **FastAPI backend**.  
   - The backend identifies the correct database collection for the request.  
   - Relevant data is retrieved from **MongoDB**.

4. **AI Response Generation:**  
   - The user's query and retrieved data are sent to **Mistral AI**.  
   - AI generates a personalized and contextual response.

5. **Response Delivery:**  
   - The response is sent back to the chat interface.  
   - Conversation history is updated in the database for continuity.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd hrm-agent-2-0
```

### 2. Create Environment File

Create a `.env` file in the project root:

```env
# Application Settings
APP_NAME=HRM Agent 2.0
ENVIRONMENT=development
DEBUG=True

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=hrm_agent_db

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=hrm-agent-index

# Mistral AI Configuration
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-medium

# JWT Authentication
SECRET_KEY=your_super_secret_jwt_key_min_32_chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# OTP Configuration
OTP_LENGTH=6
OTP_EXPIRY_MINUTES=10

# SMTP Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password
SMTP_FROM_EMAIL=your_email@gmail.com
SMTP_FROM_NAME=HRM Agent

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:8000
```

**Generate a secure SECRET_KEY:**
```bash
openssl rand -hex 32
```

### 3. Run Setup Script

The `setup.sh` script will automatically:
- ✅ Create a Python virtual environment
- ✅ Activate the environment
- ✅ Upgrade pip
- ✅ Install all dependencies from requirements.txt

```bash
# Make script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

### 4. Start the Server

```bash
# Make run script executable
chmod +x run.sh

# Start the server
./run.sh
```

The server will start on `http://127.0.0.1:8000`

### 5. Access the Application

- **Home Page**: http://127.0.0.1:8000
- **Login Page**: http://127.0.0.1:8000/login
- **Chat Interface**: http://127.0.0.1:8000/chat
- **API Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📚 API Endpoints

### Authentication Routes (`/api/auth`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | User login |
| POST | `/api/auth/verify-otp` | Verify OTP code |
| POST | `/api/auth/resend-otp` | Resend OTP |
| POST | `/api/auth/refresh` | Refresh access token |
| GET | `/api/auth/profile` | Get user profile |
| PUT | `/api/auth/profile` | Update profile |

### Chat Routes (`/api/chat`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/message` | Send chat message |
| GET | `/api/chat/history` | Get chat history |
| DELETE | `/api/chat/history` | Clear chat history |

### Mistral AI Routes (`/api/mistral`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/mistral/chat` | Chat with Mistral AI |
| POST | `/api/mistral/completion` | Get text completion |

### Web Routes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/login` | Login/signup page |
| GET | `/chat` | Chat interface |
| GET | `/health` | Health check |




