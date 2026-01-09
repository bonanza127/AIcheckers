# Repository Guidelines

## Project Structure & Module Organization
- `src/app/` contains Next.js App Router routes (e.g., `src/app/how-it-works/`, `src/app/api/`).
- `src/components/` houses shared React UI components; `src/lib/` holds frontend helpers.
- `backend/` is the FastAPI service (`backend/main.py`) with Python dependencies in `backend/requirements.txt`.
- `lib/` includes shared Python utilities used by backend/scripts.
- `public/` stores static assets; `scripts/` and `experiments/` contain ad hoc research/validation scripts and outputs.

## Build, Test, and Development Commands
- `npm run dev`: start the Next.js dev server at `http://localhost:3000`.
- `npm run build`: create a production build.
- `npm run start`: run the production build locally.
- `npm run lint`: run ESLint (Next.js core-web-vitals + TypeScript config).
- `cd backend && pip install -r requirements.txt`: install backend deps.
- `cd backend && uvicorn main:app --reload`: run the API server on `http://localhost:8000`.

## Coding Style & Naming Conventions
- TypeScript/TSX uses 2-space indentation and double quotes; follow existing file patterns.
- React components are `PascalCase` (e.g., `VipModal.tsx`); hooks/utilities are `camelCase`.
- Route folders in `src/app/` use kebab-case (e.g., `how-it-works`).

## Testing Guidelines
- There is no single test runner. Use `npm run lint` for frontend checks.
- Python validations are script-based; run targeted files in `scripts/` (e.g., `python scripts/test_jpeg_robustness.py`).
- When adding tests, prefer `test_*.py` naming to match existing scripts.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits with scopes (e.g., `feat(patrol): ...`, `fix(backend): ...`).
- PRs should include a brief summary, testing notes, and screenshots for UI changes.
- Link relevant issues and avoid committing secrets; use `.env.local` or `backend/.env` for local config.

## Security & Configuration Tips
- Environment config lives in `.env.local` and `backend/.env`; keep secrets out of git history.
