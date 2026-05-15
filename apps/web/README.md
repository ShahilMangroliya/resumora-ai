# ResumeFit — Web

Next.js 16 frontend for ResumeFit. Talks to the FastAPI backend in
`apps/api` via `POST /analyze`.

## Env

Copy `.env.example` → `.env.local` and adjust if your API is not on
`http://localhost:8000`:

```bash
cp .env.example .env.local
```

| Var | Default | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Base URL of the ResumeFit API. |

## Run

From the repo root, the recommended path is:

```bash
make dev
```

This boots the API on :8000 and the web app on :3000 together via
`honcho` (see `Procfile`).

Or directly from this directory:

```bash
npm install
npm run dev
```

Open <http://localhost:3000>.

## Build / lint

```bash
npm run build
npm run lint
```
