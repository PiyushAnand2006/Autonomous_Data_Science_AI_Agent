# AUDAS — Foundry Frontend

The **Autonomous AI Data Scientist (AUDAS)** web frontend — a complete "Foundry" re-design that
plugs into the existing FastAPI backend (`aads/api/server.py`) without any backend changes.

> Design direction: **"The Analyst's Workshop"** — a warm paper canvas, espresso-dark pipeline
> console, and furnace-gradient accents. Deliberately **no purple, no blue** "AI-gradient" clichés.
> Ships with two themes: **Dark (Fluorescent Foundry)** — the default on open, with obsidian
> `#0B0D0E` and a chartreuse `#B7F31C` accent — and **Light (Foundry)**, a warm paper palette.

---

## Design System

### Palette — "Foundry"

| Role | Light | Dark |
| --- | --- | --- |
| Page canvas | warm cream `#f4f1e8` | obsidian `#0B0D0E` |
| Cards | `#fffdf7` | `#111416` |
| Primary ink | `#231f16` | white `#ffffff` |
| Primary accent | amber → clay gradient (`#b45309 → #c2410c → #9a3412`) | chartreuse `#B7F31C` gradient |
| On-accent text | warm white `#fff8ec` | dark ink `#0C1106` |
| Success | jade `#0e7a4f` | `#6FE84C` |
| Danger | `#b3261e` | `#ff6b5e` |
| Console panel | espresso `#211a10` | raised black `#07090A` |
| Agent log tags | warm spectrum | chartreuse family + warm signals |

**Signature gradients** (light): `--grad-ember` (CTAs, champion badge, brand mark), `--grad-text`
(gradient-clipped display headings), `--grad-hairline` (section rules), and ultra-subtle ambient
washes fixed on the page background.

### Typography (tri-stack)

| Role | Font | Usage |
| --- | --- | --- |
| Display | **Fraunces** (variable serif) | page titles, card headings, model names |
| UI / Body | **Inter** | all interface copy |
| Data / Mono | **JetBrains Mono** | metrics, log tags, run IDs, file paths, table figures |

### Structure

- **Sidebar workspace shell** (desktop ≥1024px): brand, 5 primary destinations with SVG icons,
  live backend health dot, model-settings entry. Collapses to a top bar on mobile.
- **Page furniture**: eyebrow label → Fraunces title (gradient-clipped keyword) → lede → hairline rule.
- **Espresso pipeline console**: dark panel reserved exclusively for the live agent log stream —
  functional contrast, not decoration. Pulsing status dot, blinking cursor, warm per-agent tag pills.
- **Theme switch**: pill toggle at the top-right — sliding gradient knob between sun/moon;
  **dark is the default on first open**; an explicit light choice persists in `localStorage`
  (`audas_theme`) and is applied before first paint (no flash).

### Accessibility & quality gates

- No emoji icons — one hand-tuned SVG family (`src/components/Icon.jsx`), 1.8px stroke.
- Contrast ≥ 4.5:1 for body text in both themes.
- `:focus-visible` rings, `aria-*` on nav/drawers/combobox, keyboard navigation in the model
  combobox (↑↓/Enter/Esc), `role="switch"` on the theme toggle.
- `prefers-reduced-motion` honored globally; touch targets ≥ 44px; 4/8px spacing rhythm.

---

## Features (full parity)

- **Launch Pipeline** — Local (deterministic) / AI-Powered mode switching, dataset upload
  (CSV/XLSX/Parquet, drag & drop, live upload progress) or built-in sample, dataset preview
  accordion, natural-language objective, target column, SSE log stream with per-agent tags,
  run summary HUD (best model, primary metric, exports, data-health score), 1-click ZIP export.
- **Leaderboard** — top-4 exported model cards (gradient-bordered champion, per-metric readouts,
  .pkl download) + full candidate benchmark table with dynamic metric columns.
- **Visualizations** — category filter pills, responsive chart grid, keyboard-accessible lightbox
  with high-res PNG download.
- **Executive Report** — markdown renderer (tables, headers, lists, inline bold/italic/code) with
  PDF / DOCX / MD downloads.
- **Artifacts** — ZIP hero banner + 10-folder explorer with per-file size and download.
- **Settings drawer** — provider selection (OpenRouter / NVIDIA NIM / Gemini / OpenAI / Claude /
  Ollama / custom OpenAI-compatible endpoint), API keys, fetch-models, connection test,
  searchable model combobox with presets, autonomy level, processing engine, random seed.

## Run

```bash
# Terminal 1 — FastAPI backend (from the repo root)
.venv\Scripts\activate
python -m uvicorn aads.api.server:app --port 8000

# Terminal 2 — this frontend
npm install
npm run dev        # http://localhost:5173
```

The app targets `http://localhost:8000` by default. Override with:

```bash
VITE_API_URL=https://audas-backend.onrender.com npm run dev
```

### Preview without the real backend

`mock/dev-backend.mjs` is a dependency-free fixture server (sample data, scripted SSE stream,
fixtures for every page) for UI demos:

```bash
node mock/dev-backend.mjs    # serves on :8000 — trains nothing
```

## Project layout

```
src/
├── api.js                  # Backend client — same wire contract as the original
├── styles/
│   ├── tokens.css          # Design tokens (palette, gradients, type, motion, z-scale)
│   ├── global.css          # Reset, shell layout, sidebar, page furniture, a11y
│   ├── components.css      # Buttons, cards, forms, console, tables, drawer, lightbox…
│   └── dark.css            # "Fluorescent Foundry" dark theme ([data-theme='dark'])
├── components/
│   ├── Icon.jsx            # SVG icon family + brand mark
│   ├── Sidebar.jsx         # Workspace nav + backend health probe
│   ├── ThemeSwitch.jsx     # Pill light/dark toggle (top-right)
│   ├── Card.jsx / EmptyState.jsx
│   ├── FileUploader.jsx    # Upload + sample modes, drag & drop, progress
│   ├── SearchableSelect.jsx# Keyboard-navigable model combobox
│   ├── SettingsDrawer.jsx  # Provider/LLM/runtime configuration sheet
│   ├── PipelineConsole.jsx # Espresso log console + warm agent-tag parser
│   └── ModelCard.jsx
└── pages/
    ├── Dashboard.jsx       # Launch pipeline
    ├── Leaderboard.jsx
    ├── Visualizations.jsx
    ├── Report.jsx
    └── Artifacts.jsx
```
