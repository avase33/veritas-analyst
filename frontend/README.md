# Veritas — Frontend (Next.js)

Chat interface for the Veritas document analyst, built with the Next.js App
Router + TypeScript.

> The Python backend also serves a zero-build version of this chat at
> `http://localhost:8000/`. This Next.js app is the deployable frontend (Vercel).

## Develop

```bash
cd frontend
cp .env.example .env.local        # point NEXT_PUBLIC_API_URL at the backend
npm install
npm run dev                       # http://localhost:3000
```

Start the backend first:

```bash
pip install -e ".[serve]"
veritas serve                     # http://localhost:8000
```

The app proxies `/api/*` to the FastAPI backend (see `next.config.js`).

## Structure

```
app/layout.tsx     root layout + styles
app/page.tsx       page shell
components/Chat.tsx chat with citations + agent trace
lib/api.ts         typed client (/api/chat, /api/upload)
```

## Deploy

- **Frontend → Vercel**: set `NEXT_PUBLIC_API_URL` to your deployed backend URL.
- **Backend → Render/Railway**: see `../render.yaml`.
