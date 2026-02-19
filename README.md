# StackPilot 💻

> A full-stack AI-powered code generation and execution platform — built from scratch with Node.js, Express, MongoDB, JWT authentication, and the Hugging Face Inference API.

![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=flat-square&logo=node.js&logoColor=white)
![Express](https://img.shields.io/badge/Express-5.x-000000?style=flat-square&logo=express&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)
![JWT](https://img.shields.io/badge/Auth-JWT-000000?style=flat-square&logo=jsonwebtokens&logoColor=white)
![HuggingFace](https://img.shields.io/badge/AI-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## What is this?

StackPilot is a **full-stack AI coding assistant** a self-hosted Copilot with a chat interface, multi-language code execution, persistent history, and user authentication.

You type a natural language prompt. It streams back real, working code. You run it directly in the browser. Your entire session history is saved to your account.

Built this to understand how real AI-powered products actually work under the hood from LLM API integration to auth flows to safely executing user-submitted code on a server.

---

## Live Demo

> Deployed on a VPS. Server is live at: `https://api-uwu-aisoc.teamitj.tech`

---

## Features

| Feature | Details |
|---|---|
| 🤖 AI Code Generation | Streams code token-by-token using Hugging Face Inference API (Qwen2.5-72B) |
| ⚡ Real-time Streaming | Server-Sent Events (SSE) — code appears live as the model generates it |
| 🔐 JWT Authentication | Signup/login with bcrypt password hashing and 7-day JWT tokens |
| 🗄️ Persistent History | Every chat session stored in MongoDB Atlas, per user, with full message history |
| 🐍 Python Execution | Runs generated Python code in isolated temp files with a 10s timeout kill switch |
| 🟨 JavaScript Execution | Runs generated Node.js code with the same sandboxing approach |
| ✏️ In-place Code Editing | Edit any generated code block directly in the UI before running |
| 🎨 Personalized UI | Dynamic greeting changes on every visit based on time of day + user's name |
| 🎨 Theme Customization | 6 accent colors, persistent per session |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser (Client)                  │
│  Vanilla JS · SSE Stream Reader · JWT in localStorage│
└───────────────────┬─────────────────────────────────┘
                    │ HTTP / SSE
┌───────────────────▼─────────────────────────────────┐
│               Express.js Backend                     │
│                                                      │
│  /auth/signup   → bcrypt hash → MongoDB save         │
│  /auth/login    → bcrypt compare → JWT sign          │
│  /chat          → HF API stream → SSE to client      │
│                   → save messages to MongoDB         │
│  /run           → write tmp file → spawn process     │
│                   → capture stdout/stderr → respond  │
│  /history       → MongoDB query → return chats       │
└──────┬──────────────────────────┬────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  MongoDB    │          │  Hugging Face   │
│  Atlas      │          │  Inference API  │
│  (Users +   │          │  Qwen2.5-72B    │
│   Chats)    │          └─────────────────┘
└─────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Runtime | Node.js 18+ | Non-blocking I/O — essential for SSE streaming |
| Framework | Express.js 5 | Minimal, composable, industry standard |
| Database | MongoDB + Mongoose | Flexible schema fits chat message structures well |
| Auth | JWT + bcryptjs | Stateless auth — scales horizontally, no session store needed |
| AI | Hugging Face Inference API | Access to open-weight models without hosting GPUs |
| Streaming | Server-Sent Events | One-directional stream from server — lighter than WebSockets for this use case |
| Code Execution | child_process.spawn | Async, non-blocking, timeout-enforced execution |
| Frontend | Vanilla JS | No framework overhead — good exercise in DOM fundamentals |
| Syntax Highlighting | Prism.js | Lightweight, supports 200+ languages |
| Fonts | JetBrains Mono + Outfit | Developer-optimized monospace + clean UI sans-serif |

---

## API Reference

### Auth

```http
POST /auth/signup
Content-Type: application/json

{ "name": "Mansha", "email": "m@example.com", "password": "secret123" }

→ 201 { token, user: { id, name, email } }
```

```http
POST /auth/login
Content-Type: application/json

{ "email": "m@example.com", "password": "secret123" }

→ 200 { token, user: { id, name, email } }
```

```http
GET /auth/me
Authorization: Bearer <token>

→ 200 { user: { id, name, email, createdAt } }
```

### Chat (Protected)

```http
POST /chat
Authorization: Bearer <token>
Content-Type: application/json

{ "message": "fibonacci sequence", "language": "python", "chatId": "optional" }

→ text/event-stream
  data: {"chatId": "..."}
  data: {"content": "def "}
  data: {"content": "fibonacci"}
  ...
  data: [DONE]
```

### Code Execution (Protected)

```http
POST /run
Authorization: Bearer <token>
Content-Type: application/json

{ "code": "print('hello')", "language": "python" }

→ 200 { "output": "hello\n" }
→ 500 { "error": "NameError: name 'x' is not defined" }
```

### History (Protected)

```http
GET  /history          → list of all user's chats (title, language, timestamps)
GET  /history/:id      → full chat with all messages
POST /history          → create new empty chat session
DELETE /history/:id    → delete a chat
```

---

## Security Design

Running user-submitted code on a server is a real security problem. Here's what this project implements and what it doesn't (yet):

**Implemented:**
- ⏱️ **Execution timeout** — `child_process` is killed after 10 seconds, preventing infinite loops from hanging the server
- 🗑️ **Temp file cleanup** — files are deleted immediately after execution finishes
- 🔑 **JWT middleware** — `/chat` and `/run` are protected routes, unauthenticated requests get 401
- 🔒 **bcrypt hashing** — passwords are never stored in plaintext (12 salt rounds)
- 📁 **Isolated working directory** — spawned processes run in `/tmp`, not the project root

**Known limitations (for production, would add):**
- Docker container sandboxing per execution
- Memory limits via `ulimit`
- Network isolation for spawned processes
- Rate limiting per user on `/run`

> ⚠️ This project is intended for educational and portfolio use. The README says it, the code has comments about it, and any production deployment would need the Docker sandbox layer.

---

## Database Schema

```javascript
// User
{
  name:      String (required),
  email:     String (unique, lowercase),
  password:  String (bcrypt hashed, never returned in API responses),
  createdAt: Date
}

// Chat
{
  userId:   ObjectId → User,
  title:    String (first 40 chars of first message),
  language: "python" | "javascript",
  messages: [
    {
      role:      "user" | "assistant",
      content:   String,
      code:      String (extracted clean code, no fences),
      language:  String,
      timestamp: Date
    }
  ],
  createdAt: Date,
  updatedAt: Date
}
```

---

## Setup

### Prerequisites

- Node.js 18+
- Python 3.x (for Python execution)
- Node.js (for JavaScript execution — it's already installed if you're running this)
- MongoDB Atlas account (free tier works)

### Installation

```bash
# Clone
git clone https://github.com/yourusername/stackpilot
cd stackpilot

# Install dependencies
npm install

# Configure environment
cp .env.template .env
# Fill in your values (see below)

# Run
node server.js
```

### Environment Variables

```env
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/stackpilot
JWT_SECRET=any_long_random_string_at_least_32_chars
PORT=9000
```

**Get your keys:**
- HuggingFace API key → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- MongoDB URI → [mongodb.com/atlas](https://mongodb.com/atlas) → Free cluster → Connect → Drivers

---

## Project Structure

```
stackpilot/
├── server.js          # Express app — auth, chat, execution, history routes
├── index.html         # Single-page frontend — all JS inline, no framework
├── package.json
├── .env               # Never committed
├── .gitignore
├── tmp/               # Temp execution files (auto-created, auto-cleaned)
└── README.md
```

No `/src`, no `/routes`, no `/controllers` — kept intentionally flat. For a project this size, folder structure adds navigation overhead with no real benefit. If this scaled to 10+ route files, I'd split it.

---

## What I Learned Building This

**Streaming is harder than it looks.** Getting SSE to work correctly — parsing chunks, handling `[DONE]`, rebuilding code incrementally in the DOM without flicker — took way more iteration than expected. The browser's `ReadableStream` API has quirks when chunks arrive mid-JSON.

**`e.currentTarget` dies after `await`.** Spent a solid hour debugging a page refresh caused by a `null` reference on a button element. `currentTarget` is set to null by the browser once the click handler's synchronous execution ends — so in an `async` function, anything after the first `await` has a null `currentTarget`. Fixed with event delegation.

**JWT is stateless for a reason.** Having no session store means you can run multiple instances of this server without them needing to share state. That's why JWTs are popular in microservice architectures — a downstream service can verify the token without calling back to an auth service.

**System prompts matter as much as the model.** The same Qwen2.5-72B model would return code snippets wrapped in markdown with a vague prompt, and return complete runnable files with a specific one. Prompt engineering isn't magic — it's just being precise about what you want.

---

## Roadmap

- [ ] Docker sandbox for code execution (proper isolation)
- [ ] Rate limiting per user (`express-rate-limit`)
- [ ] Model selector in UI (swap between models)
- [ ] Export chat history as `.md` or `.pdf`
- [ ] C++ and Java execution support
- [ ] WebSocket upgrade for bidirectional chat (future multi-agent use case)

---

