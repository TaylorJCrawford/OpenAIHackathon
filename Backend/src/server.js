import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import Joi from 'joi';
import Mustache from 'mustache';
import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';

// --- Environment validation ---
const envSchema = Joi.object({
  OPENAI_API_KEY: Joi.string().min(20).required(),
  PORT: Joi.number().default(3000),
  NODE_ENV: Joi.string().valid('development','production','test').default('development'),
  CORS_ORIGIN: Joi.string().default('*'),
  RATE_LIMIT_WINDOW_MS: Joi.number().default(60_000),
  RATE_LIMIT_MAX: Joi.number().default(60),
}).unknown(true);

const { value: env, error: envErr } = envSchema.validate(process.env);
if (envErr) {
  console.error('❌ Invalid environment:', envErr.message);
  process.exit(1);
}

// --- OpenAI client ---
const client = new OpenAI({ apiKey: env.OPENAI_API_KEY });

// --- Model presets (override with request fields) ---
const MODEL_PRESETS = {
  'gpt-5': {
    model: 'gpt-5',
    temperature: 0.7,
    max_output_tokens: 1024
  },
  'gpt-5-mini': {
    model: 'gpt-5-mini',
    temperature: 0.5,
    max_output_tokens: 768
  },
  'gpt-5-chat-latest': {
    model: 'gpt-5-chat-latest',
    temperature: 0.7,
    max_output_tokens: 1024
  }
};

// --- Prompt templates (system + optional user prelude) ---
const TEMPLATES = {
  general: {
    system: `You are a helpful, concise assistant. Prefer clear, direct answers.`,
    userPrelude: ''
  },
  coding: {
    system: `You are a senior software engineer. Provide runnable examples, point out edge cases, and include time/space tradeoffs when relevant.`,
    userPrelude: ''
  },
  support: {
    system: `You are a friendly customer support agent. Be empathetic, clarify next steps, and summarize actions.`,
    userPrelude: ''
  }
};

// --- Request validation ---
const bodySchema = Joi.object({
  role: Joi.string().required(),
  prompt: Joi.string().required()
});

// --- Helper function for chat response ---
async function getChatResponse(role, prompt) {
  // [Building Context] Load context from local JSON file
  let contextArr = [];
  try {
    const contextPath = path.join(process.cwd(), 'data', 'context.json');
    const contextRaw = await fs.readFile(contextPath, 'utf-8');
    contextArr = JSON.parse(contextRaw);
  } catch (e) {
    contextArr = [];
  }

  // [GuardRailPrompt] Load guardrail prompt from local JSON file
  let guardrailPrompt = '';
  try {
    const guardrailPath = path.join(process.cwd(), 'data', 'guardrail.json');
    const guardrailRaw = await fs.readFile(guardrailPath, 'utf-8');
    guardrailPrompt = JSON.parse(guardrailRaw).prompt || '';
  } catch (e) {
    guardrailPrompt = '';
  }

  // Combine context, guardrail, and user prompt
  const contextString = contextArr.map(c => c.info).join('\n');
  const fullPrompt = [guardrailPrompt, contextString, prompt].filter(Boolean).join('\n---\n');

  // Call OpenAI
  const model = 'gpt-5';
  const preset = MODEL_PRESETS[model];
  const input = [{ role, content: [{ type: 'text', text: fullPrompt }] }];
  try {
    const r = await client.responses.create({
      model: preset.model,
      temperature: preset.temperature,
      max_output_tokens: preset.max_output_tokens,
      input
    });
    return r.output_text ?? (r.output?.map(o => o.content?.map(c => c.text).join('')).join('\n') || '');
  } catch (e) {
    return '[OpenAI error: ' + (e?.error?.message || e.message || 'Unknown error') + ']';
  }
}

// --- Express app ---
const app = express();
app.use(cors({ origin: env.CORS_ORIGIN === '*' ? true : env.CORS_ORIGIN.split(',') }));
app.use(helmet());
app.use(express.json({ limit: '1mb' }));
app.set('trust proxy', 1);
app.use(rateLimit({ windowMs: Number(env.RATE_LIMIT_WINDOW_MS), max: Number(env.RATE_LIMIT_MAX) }));

app.get('/health', (_req, res) => res.json({ ok: true }));

// Single POST route
app.post('/v1/chat', async (req, res) => {
  const { value, error } = bodySchema.validate(req.body, { abortEarly: false, stripUnknown: true });
  if (error) {
    return res.status(400).json({ ok: false, error: { type: 'BadRequest', message: error.message } });
  }

  // Call the helper method
  const output_text = await getChatResponse(value.role, value.prompt);

  res.json({ ok: true, model: 'gpt-5', output_text, usage: null, raw: null });
});

app.use((err, _req, res, _next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ ok: false, error: { type: 'ServerError', message: 'Internal server error' } });
});

const server = app.listen(env.PORT, () => {
  console.log(`🚀 gpt5-rest listening on http://localhost:${env.PORT}`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`❌ Port ${env.PORT} is already in use. Please set a different PORT environment variable or stop the process using this port.`);
    process.exit(1);
  } else {
    throw err;
  }
});
