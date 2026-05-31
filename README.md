# AFAD Earthquake Learning Dashboard

A local research dashboard that ingests AFAD earthquake catalog data, trains a TensorFlow.js sequence model, and updates earthquake forecast signals over time.

## What it does

- Loads and stores earthquake catalog data locally
- Trains a GRU-based multi-head model on recent earthquake sequences
- Tracks region, location, wait-time, magnitude, depth, and large-event signals
- Shows a live dashboard for model state, recent comparisons, and forecast summaries

## Important note

This project is **research-oriented only**.

Earthquakes cannot be predicted reliably in a way that should be used for public safety, emergency response, or life-critical decisions. The outputs of this application must not be treated as an official warning system.

## Tech stack

- React
- Vite
- Express
- Socket.IO
- TensorFlow.js

## Local development

Install dependencies:

```bash
npm install
```

Run the app:

```bash
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Environment variables

You can override the main catalog start date and fetch interval:

```powershell
$env:LEARNING_START_DATE="2024-05-12 00:00:00"
$env:FETCH_INTERVAL_MS="120000"
npm run dev
```

## Project structure

- `src/` - React frontend
- `server.mjs` - backend, training pipeline, AFAD ingestion, and API
- `data/` - local runtime data and model state
- `shared/` - shared utilities
- `scripts/` - helper scripts

## Repository notes

Large local runtime files such as catalog snapshots, model weights, logs, and generated artifacts are intentionally excluded from version control.
