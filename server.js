const express   = require('express');
const cors      = require('cors');
const mongoose  = require('mongoose');
const bcrypt    = require('bcryptjs');
const jwt       = require('jsonwebtoken');
const { HfInference } = require('@huggingface/inference');
const { spawn } = require('child_process');
const fs        = require('fs').promises;
const path      = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

/* ═══════════════════════════════════════
   DATABASE CONNECTION
═══════════════════════════════════════ */
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB error:', err));

/* ═══════════════════════════════════════
   MODELS
═══════════════════════════════════════ */

// User
const userSchema = new mongoose.Schema({
  name:      { type: String, required: true, trim: true },
  email:     { type: String, required: true, unique: true, lowercase: true, trim: true },
  password:  { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

const User = mongoose.model('User', userSchema);

// Chat / Session
const messageSchema = new mongoose.Schema({
  role:      { type: String, enum: ['user', 'assistant'] },
  content:   { type: String },
  code:      { type: String, default: '' },
  language:  { type: String, default: 'python' },
  timestamp: { type: Date, default: Date.now }
});

const chatSchema = new mongoose.Schema({
  userId:    { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  title:     { type: String, default: 'New Chat' },
  language:  { type: String, default: 'python' },
  messages:  [messageSchema],
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

const Chat = mongoose.model('Chat', chatSchema);

/* ═══════════════════════════════════════
   AUTH MIDDLEWARE
═══════════════════════════════════════ */
function verifyToken(req, res, next) {
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No token provided' });
  }
  try {
    const decoded = jwt.verify(auth.split(' ')[1], process.env.JWT_SECRET);
    req.userId = decoded.id;
    next();
  } catch {
    res.status(401).json({ error: 'Invalid or expired token' });
  }
}

/* ═══════════════════════════════════════
   AUTH ROUTES
═══════════════════════════════════════ */

// Signup
app.post('/auth/signup', async (req, res) => {
  try {
    const { name, email, password } = req.body;

    if (!name || !email || !password)
      return res.status(400).json({ error: 'All fields are required' });

    if (password.length < 6)
      return res.status(400).json({ error: 'Password must be at least 6 characters' });

    const exists = await User.findOne({ email });
    if (exists)
      return res.status(409).json({ error: 'Email already registered' });

    const user = await User.create({ name, email, password });

    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '7d' });

    res.status(201).json({
      token,
      user: { id: user._id, name: user.name, email: user.email }
    });

  } catch (err) {
    console.error('Signup error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Login
app.post('/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password)
      return res.status(400).json({ error: 'Email and password are required' });

    const user = await User.findOne({ email });
    if (!user)
      return res.status(401).json({ error: 'Invalid email or password' });

    const match = await bcrypt.compare(password, user.password);
    if (!match)
      return res.status(401).json({ error: 'Invalid email or password' });

    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '7d' });

    res.json({
      token,
      user: { id: user._id, name: user.name, email: user.email }
    });

  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Verify token / get current user
app.get('/auth/me', verifyToken, async (req, res) => {
  try {
    const user = await User.findById(req.userId).select('-password');
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json({ user });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ═══════════════════════════════════════
   HISTORY ROUTES
═══════════════════════════════════════ */

// Get all chats for user
app.get('/history', verifyToken, async (req, res) => {
  try {
    const chats = await Chat.find({ userId: req.userId })
      .select('title language createdAt updatedAt')
      .sort({ updatedAt: -1 })
      .limit(50);
    res.json({ chats });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get single chat with messages
app.get('/history/:id', verifyToken, async (req, res) => {
  try {
    const chat = await Chat.findOne({ _id: req.params.id, userId: req.userId });
    if (!chat) return res.status(404).json({ error: 'Chat not found' });
    res.json({ chat });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Create new chat session
app.post('/history', verifyToken, async (req, res) => {
  try {
    const { title = 'New Chat', language = 'python' } = req.body;
    const chat = await Chat.create({ userId: req.userId, title, language, messages: [] });
    res.status(201).json({ chat });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete a chat
app.delete('/history/:id', verifyToken, async (req, res) => {
  try {
    await Chat.deleteOne({ _id: req.params.id, userId: req.userId });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ═══════════════════════════════════════
   HF + CHAT GENERATION
═══════════════════════════════════════ */
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);
console.log('HF KEY EXISTS:', !!process.env.HUGGINGFACE_API_KEY);

const SYSTEM_PROMPT = `You are an expert coding assistant. 
Generate COMPLETE, fully working code that runs without modification.

STRICT RULES:
1. Always output the FULL code — never a snippet, never partial code.
2. For JavaScript: Always output a complete HTML file with <!DOCTYPE html>, <head>, <body>, <style>, and <script> tags all in one file, so it can be opened in a browser and work immediately.
3. For Python: Always output a complete runnable Python script with all imports and logic.
4. Include helpful comments so the code is readable.
5. The code must actually work when run — test your logic mentally before outputting.
6. No markdown fences, no backticks, no explanations outside of code comments.
7. Never give a snippet. If asked for a button, build the full HTML page around it.`;

app.post('/chat', verifyToken, async (req, res) => {
  try {
    const { message, language = 'python', chatId } = req.body;
    if (!message) return res.status(400).json({ error: 'No message provided' });

    // Find or create chat session
    let chat;
    if (chatId) {
      chat = await Chat.findOne({ _id: chatId, userId: req.userId });
    }
    if (!chat) {
      chat = await Chat.create({
        userId: req.userId,
        title: message.slice(0, 40),
        language,
        messages: []
      });
    }

    // Save user message
    chat.messages.push({ role: 'user', content: message, language });
    chat.updatedAt = new Date();
    await chat.save();

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Send chatId so frontend can track session
    res.write(`data: ${JSON.stringify({ chatId: chat._id.toString() })}\n\n`);

    const stream = hf.chatCompletionStream({
      model: 'Qwen/Qwen2.5-72B-Instruct',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: `Write simple ${language} code for: ${message}` }
      ],
      temperature: 0.3,
      max_tokens: 2048
    });

    let fullCode = '';

    for await (const chunk of stream) {
      const content = chunk?.choices?.[0]?.delta?.content;
      if (!content) continue;
      fullCode += content;
      res.write(`data: ${JSON.stringify({ content })}\n\n`);
    }

    // Save assistant response
    const cleanCode = fullCode.replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '').trim();
    chat.messages.push({ role: 'assistant', content: message, code: cleanCode, language });
    chat.updatedAt = new Date();
    await chat.save();

    res.write('data: [DONE]\n\n');
    res.end();

  } catch (err) {
    console.error('Chat error:', err);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

/* ═══════════════════════════════════════
   RUN CODE
═══════════════════════════════════════ */
const LANG_CONFIG = {
  python:     { ext: '.py',  cmd: 'python', args: f => [f] },
  javascript: { ext: '.js',  cmd: 'node',   args: f => [f] }
};

app.post('/run', verifyToken, async (req, res) => {
  const { code, language = 'python' } = req.body;
  if (!code) return res.status(400).json({ error: 'No code provided' });

  const lang = LANG_CONFIG[language];
  if (!lang) return res.status(400).json({ error: `Unsupported language: ${language}` });

  try {
    const tmpDir = path.join(__dirname, 'tmp');
    await fs.mkdir(tmpDir, { recursive: true });
    const tmpFile = path.join(tmpDir, `code_${Date.now()}${lang.ext}`);
    await fs.writeFile(tmpFile, code);

    const output = await new Promise((resolve, reject) => {
      let stdout = '', stderr = '';
      const proc = spawn(lang.cmd, lang.args(tmpFile), { cwd: __dirname });

      const timer = setTimeout(() => { proc.kill(); reject(new Error('Execution timed out (10s)')); }, 10000);

      proc.stdout.on('data', d => stdout += d.toString());
      proc.stderr.on('data', d => stderr += d.toString());
      proc.on('close', code => {
        clearTimeout(timer);
        fs.unlink(tmpFile).catch(() => {});
        if (code !== 0) reject(new Error(stderr || 'Execution failed'));
        else resolve(stdout);
      });
    });

    res.json({ output });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ═══════════════════════════════════════
   SERVE FRONTEND
═══════════════════════════════════════ */
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')));

const PORT = process.env.PORT || 9000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));