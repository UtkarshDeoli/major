# Orbit — AI-Powered Study Platform

Orbit is an AI-powered study platform built for Indian coaching centers, tuition classes, schools, and independent students. It combines document-based RAG chat, mock-test generation, flashcards, study planning, and subscription billing into one sellable product.

## Architecture

- **Frontend**: Next.js 16 + TypeScript + Tailwind CSS (`Frontend/`)
- **Backend**: FastAPI + Motor/MongoDB + Gemini AI (`Backend/`)
- **Payments**: Razorpay (India-first subscriptions + org seat licenses)
- **AI model**: Gemini 2.5 Flash

## Quick start

### Backend

```bash
cd Backend
cp .env.example .env
# Fill in MONGODB_URL, GEMINI_API_KEY, SECRET_KEY, and Razorpay keys
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

Run tests:

```bash
pytest tests/
```

### Frontend

```bash
cd Frontend
npm install
# Set NEXT_PUBLIC_API_URL=http://localhost:8001 in .env.local
npm run dev
```

Production build:

```bash
npm run build
npm run lint
```

## Key features

- **Subscriptions & billing**: Starter / Pro / Premium plans with Razorpay checkout, usage meters, invoices, and plan enforcement.
- **Multi-tenant org seats**: Coaching centers can buy seat licenses and invite teachers/students with invite codes.
- **AI tutor chat**: Document-grounded chat with Socratic step-by-step explanations, multilingual responses, and image input.
- **Mock tests**: Adaptive difficulty based on past performance, AI rubric grading, teacher-marked mode.
- **Flashcards & summaries**: AI-generated decks and study materials from uploaded PDFs.
- **Teacher dashboard**: Managed students, at-risk alerts, recommended focus topics.
- **Focus mode**: Pomodoro-style timer with session tracking.
- **Study planner**: AI-generated weekly plans tied to exam dates and weak topics.
- **Admin panel**: User/org/subscription management and revenue analytics.

## Environment variables

### Backend `.env`

```env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=orbit
GEMINI_API_KEY=...
SECRET_KEY=...
FRONTEND_URL=http://localhost:3000
RAZORPAY_KEY_ID=...
RAZORPAY_KEY_SECRET=...
RAZORPAY_WEBHOOK_SECRET=...
```

### Frontend `.env.local`

```env
NEXT_PUBLIC_API_URL=http://localhost:8001
```

## Deployment notes

- Use a production MongoDB cluster (Atlas or self-hosted replica set).
- Switch slowapi storage from `memory://` to Redis for multi-worker deployments.
- Serve the frontend from Vercel/Netlify/self-hosted Node.js and point `NEXT_PUBLIC_API_URL` to the backend.
- Configure Razorpay webhooks to hit `https://your-api.com/webhooks/razorpay`.
- Update `FRONTEND_URL` and tighten CSP in `Frontend/next.config.js` for your domain.

## Plan enforcement

The backend enforces per-plan limits before expensive AI calls:

| Resource | Starter | Pro | Premium |
|---|---|---|---|
| Mock tests / month | 3 | 50 | Unlimited |
| Flashcards / month | 50 | 500 | Unlimited |
| AI summaries / month | 5 | 50 | Unlimited |
| Chat messages / month | 100 | 1,000 | Unlimited |
| Document storage | 50 MB | 1 GB | 10 GB |
| Classes / batches | 1 | 10 | Unlimited |

Exceeded limits return HTTP 402 with an upgrade payload.

## Tests

- Backend: `pytest tests/` (69 tests passing)
- Frontend: `npm run lint` + `npm run build`

## Roadmap status

- ✅ Phase 1 — Sellable backbone (billing, orgs, admin, plan enforcement, rate limits)
- ✅ Phase 2a — AI tutoring (Socratic, rubric grading, adaptive tests, alerts, multilingual)
- ✅ Phase 2b/3 — Study experience + production hardening (focus mode, image chat, study planner, security headers, error boundaries)
