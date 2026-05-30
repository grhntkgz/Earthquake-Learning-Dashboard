import express from "express";
import { createServer } from "node:http";
import { Server } from "socket.io";
import fs from "node:fs/promises";
import { writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import * as tf from "@tensorflow/tfjs";
import { REGION_BOUNDS, REGION_ORDER, classifyRegion } from "./shared/regions.js";

function pad2(value) {
  return String(value).padStart(2, "0");
}

function defaultStartDateTurkey(yearsAgo) {
  const now = new Date();
  const yearsAgoDate = new Date(Date.UTC(
    now.getUTCFullYear() - yearsAgo,
    now.getUTCMonth(),
    now.getUTCDate(),
    0,
    0,
    0,
  ));
  return `${yearsAgoDate.getUTCFullYear()}-${pad2(yearsAgoDate.getUTCMonth() + 1)}-${pad2(yearsAgoDate.getUTCDate())} 00:00:00`;
}

function yearsCoveredByStartDate(startDateTurkey) {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(startDateTurkey || "");
  if (!match) return null;
  const [, year, month, day] = match;
  const start = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day), 0, 0, 0));
  const now = new Date();
  const diffYears = (now.getTime() - start.getTime()) / (365.25 * 24 * 60 * 60 * 1000);
  return Math.max(1, Math.round(diffYears));
}

const PORT = Number(process.env.PORT || 3030);
const VITE_PORT = Number(process.env.VITE_PORT || 5173);
const AFAD_URL = "https://deprem.afad.gov.tr/apiv2/event/filter";
const START_DATE_TURKEY = process.env.LEARNING_START_DATE || defaultStartDateTurkey(2);
const LARGE_EVENT_START_DATE_TURKEY = process.env.LARGE_EVENT_START_DATE || defaultStartDateTurkey(5);
const MAIN_CATALOG_YEARS = yearsCoveredByStartDate(START_DATE_TURKEY) || 2;
const LARGE_EVENT_CATALOG_YEARS = yearsCoveredByStartDate(LARGE_EVENT_START_DATE_TURKEY) || 5;
const FETCH_INTERVAL_MS = Number(process.env.FETCH_INTERVAL_MS || 120_000);
const FETCH_TIMEOUT_MS = Number(process.env.FETCH_TIMEOUT_MS || 60_000);
const LOOKBACK = 16;
const SUMMARY_WINDOWS = [24, 48, 96];
const MOTIF_HISTORY_WINDOW = 32;
const MOTIF_FEATURE_SIZE = 6;
const TRIGGER_HISTORY_WINDOW = 256;
const TRANSITION_RECENT_WINDOW = 32;
const PRESSURE_RECENT_WINDOW = 24;
const INCREMENTAL_MAIN_EVENT_LIMIT = 6000;
const INCREMENTAL_LARGE_EVENT_LIMIT = 24000;
const INITIAL_BOOTSTRAP_MAIN_EPOCHS = 4;
const INITIAL_BOOTSTRAP_LARGE_EVENT_EPOCHS = 2;
const GRID_ROWS = 3;
const GRID_COLS = 3;
const GRID_CELL_COUNT = GRID_ROWS * GRID_COLS;
const EVENT_BASE_FEATURE_NAMES = [
  "latitude",
  "longitude",
  "depth",
  "magnitude",
  "hour_sin",
  "hour_cos",
  "delta_hours",
  "distance_from_previous",
  "mean_distance_to_recent3",
  "mean_distance_to_recent5",
  "centroid_distance3",
  "centroid_distance5",
  "same_region_as_previous",
  "recent_zone_dominance",
  "transition_surprise",
  "transition_probability",
  "trigger_persistence_score",
  "activation_lift",
  "source_pressure",
  "target_sensitivity",
  "mean_wait_short",
  "mean_wait_medium",
  "wait_compression",
  "since_last_mag25",
  "since_last_mag35",
  "mean_magnitude_medium",
  "std_magnitude_medium",
  "cluster_spread8km",
];
const EVENT_REGION_FEATURE_NAMES = REGION_ORDER.map((region) => `region_${region}`);
const EVENT_FEATURE_NAMES = [...EVENT_BASE_FEATURE_NAMES, ...EVENT_REGION_FEATURE_NAMES];
const EVENT_FEATURE_SIZE = EVENT_FEATURE_NAMES.length;
const SUMMARY_FEATURE_NAMES = [
  "mean_latitude",
  "mean_longitude",
  "std_latitude",
  "std_longitude",
  "mean_depth",
  "mean_magnitude",
  "window_fill_ratio",
  "event_density",
  "mean_distance_from_latest",
  "mean_inter_event_distance",
  "std_inter_event_distance",
  "mean_inter_event_hours",
  "std_inter_event_hours",
  "latest_to_centroid_distance",
  "mean_distance_to_centroid",
  "zone_switch_rate",
  "region_dominance",
  "recent_six_hours_ratio",
];
const SUMMARY_FEATURE_SIZE = SUMMARY_FEATURE_NAMES.length;
const SUMMARY_INPUT_SIZE = SUMMARY_WINDOWS.length * SUMMARY_FEATURE_SIZE;
const MOTIF_FEATURE_NAMES = [
  "motif4_repeat_rate",
  "motif5_repeat_rate",
  "motif6_repeat_rate",
  "motif4_seen_before",
  "motif5_seen_before",
  "motif6_seen_before",
];
const DATA_DIR = path.join(process.cwd(), "data");
const EVENTS_FILE = path.join(DATA_DIR, "events.json");
const LARGE_EVENT_EVENTS_FILE = path.join(DATA_DIR, "large-event-events.json");
const MODEL_WEIGHTS_FILE = path.join(DATA_DIR, "model-weights.json");
const PREDICTION_STATE_FILE = path.join(DATA_DIR, "prediction-state.json");
const RUNTIME_STATE_FILE = path.join(DATA_DIR, "runtime-state.json");
const MODEL_VERSION = 25;
const MATCH_HISTORY_SCHEMA_VERSION = 2;
const LARGE_EVENT_SIGNAL_SCHEMA_VERSION = 2;
const LARGE_EVENT_MAGNITUDE_THRESHOLD = 4.0;
const LARGE_EVENT_LOOKAHEAD_HOURS = 24;
const WAIT_HOURS_MAX = 168;
const WAIT_MINUTES_MAX = WAIT_HOURS_MAX * 60;
const MIN_PREDICTED_WAIT_MINUTES = 0.5;
const LOCATION_LOSS_REFERENCE_KM = 250;
const DEPTH_LOSS_REFERENCE_KM = 10;
const MAGNITUDE_LOSS_REFERENCE_ML = 1;
const WAIT_LOSS_REFERENCE_MINUTES = 30;
const DEFAULT_CONFIDENCE_THRESHOLD = 0.45;
const MAX_CONFIDENCE_THRESHOLD = 0.8;
const MIN_CONFIDENCE_THRESHOLD = 0.3;
const HIGH_CONFIDENCE_OFFSET = 0.18;
const MIN_TARGET_RECALL = 0.55;
const BASE_LOSS_WEIGHTS = Object.freeze({
  location_head: 0.75,
  grid_head: 0.7,
  scalar_head: 0.9,
  wait_head: 1.0,
  region_head: 1.15,
  large_event_head: 0.65,
});
const LOSS_REBALANCE_ALPHA = 0.35;
const LOSS_WEIGHT_FACTOR_MIN = 0.7;
const LOSS_WEIGHT_FACTOR_MAX = 1.4;
const REGION_BALANCE_ALPHA = 0.5;
const REGION_BALANCE_WEIGHT_MIN = 0.65;
const REGION_BALANCE_WEIGHT_MAX = 2.5;
const REGION_ADAPTIVE_SMOOTHING = 0.35;
const REGION_PERFORMANCE_MIN_SAMPLES = 6;
const REGION_PERFORMANCE_DISTANCE_CLAMP_MIN = 0.8;
const REGION_PERFORMANCE_DISTANCE_CLAMP_MAX = 1.8;
const REGION_PERFORMANCE_RATE_CLAMP_MIN = 0.8;
const REGION_PERFORMANCE_RATE_CLAMP_MAX = 1.8;
let sessionStartedAt = new Date();

const BOUNDS = {
  latMin: 34,
  latMax: 43.8,
  lonMin: 24,
  lonMax: 46,
  depthMax: 80,
  magMax: 8,
};

const state = {
  events: [],
  seen: new Set(),
  largeEventTrainingEvents: [],
  largeEventTrainingSeen: new Set(),
  metrics: {
    trainedSamples: 0,
    epochs: 0,
    lastLoss: null,
    lastLocationLoss: null,
    lastGridLoss: null,
    lastScalarLoss: null,
    lastWaitLoss: null,
    lastRegionLoss: null,
    lastLargeEventLoss: null,
    meanDistanceKm: null,
    medianDistanceKm: null,
    meanWaitErrorMinutes: null,
    freshEvents: 0,
    fetchCount: 0,
    lastFetchAt: null,
    lastTrainingAt: null,
    sessionStartedAt: sessionStartedAt.toISOString(),
    tableMetricsResetAt: null,
    confusionMetricsResetAt: null,
    totalLearningMs: 0,
    totalEpochs: 0,
    totalTrainedSamples: 0,
    lastRunTrainedSamples: 0,
    totalTrainingRunCount: 0,
    trainingRunCount: 0,
    totalTrainingMs: 0,
    lastTrainingMs: null,
    currentLossWeights: { ...BASE_LOSS_WEIGHTS },
    regionClassWeights: Object.fromEntries(REGION_ORDER.map((region) => [region, 1])),
    largeEventPositiveRate: null,
    largeEventPositiveWeight: 1,
    missedMatchCount: 0,
    lastRegionCombinationAt: null,
    modelLoadedAt: null,
    modelSavedAt: null,
    modelMemory: "new",
    trainingProgress: {
      active: false,
      mode: null,
      phase: null,
      phaseLabel: null,
      overallEpoch: 0,
      overallEpochs: 0,
      currentEpoch: 0,
      phaseEpochs: 0,
      currentBatch: 0,
      totalBatches: 0,
      completedUnits: 0,
      totalUnits: 0,
      progressRatio: 0,
      etaMs: null,
      startedAt: null,
      updatedAt: null,
    },
    lastAction: null,
    status: "starting",
  },
  predictions: [],
  previousPrediction: null,
  lastMatch: null,
  recentMatches: [],
  comparisonHistory: [],
  largeEventSignals: [],
  regionCombinationPatterns: [],
  showMapHistory: true,
};

let model;
let io;
let persistedLearningMs = 0;
let trainingInProgress = false;
let trainingStartPending = false;
let localCatalogTrainingActive = false;
let afadCycleSuspended = false;
let resetPending = false;
let largeEventPerformanceCache = { key: null, value: null };

await fs.mkdir(DATA_DIR, { recursive: true });

function currentLearningMs() {
  return persistedLearningMs + (Date.now() - sessionStartedAt.getTime());
}

function runtimePayload() {
  return {
    savedAt: new Date().toISOString(),
    totalLearningMs: currentLearningMs(),
    totalEpochs: state.metrics.totalEpochs,
    totalTrainedSamples: state.metrics.totalTrainedSamples,
    lastRunTrainedSamples: state.metrics.lastRunTrainedSamples,
    totalTrainingRunCount: state.metrics.totalTrainingRunCount,
    totalTrainingMs: state.metrics.totalTrainingMs,
    lastLoss: state.metrics.lastLoss,
    lastLocationLoss: state.metrics.lastLocationLoss,
    lastGridLoss: state.metrics.lastGridLoss,
    lastScalarLoss: state.metrics.lastScalarLoss,
    lastWaitLoss: state.metrics.lastWaitLoss,
    lastRegionLoss: state.metrics.lastRegionLoss,
    lastLargeEventLoss: state.metrics.lastLargeEventLoss,
    currentLossWeights: state.metrics.currentLossWeights,
    regionClassWeights: state.metrics.regionClassWeights,
    largeEventPositiveRate: state.metrics.largeEventPositiveRate,
    largeEventPositiveWeight: state.metrics.largeEventPositiveWeight,
    missedMatchCount: state.metrics.missedMatchCount,
    lastRegionCombinationAt: state.metrics.lastRegionCombinationAt,
    meanDistanceKm: state.metrics.meanDistanceKm,
    medianDistanceKm: state.metrics.medianDistanceKm,
    meanWaitErrorMinutes: state.metrics.meanWaitErrorMinutes,
    lastTrainingAt: state.metrics.lastTrainingAt,
    lastTrainingMs: state.metrics.lastTrainingMs,
    trainingProgress: state.metrics.trainingProgress,
    lastSessionStartedAt: sessionStartedAt.toISOString(),
    tableMetricsResetAt: state.metrics.tableMetricsResetAt,
    confusionMetricsResetAt: state.metrics.confusionMetricsResetAt,
  };
}

async function loadRuntimeState() {
  try {
    const memory = JSON.parse(await fs.readFile(RUNTIME_STATE_FILE, "utf8"));
    persistedLearningMs = Number(memory.totalLearningMs || 0);
    state.metrics.totalLearningMs = currentLearningMs();
    state.metrics.totalEpochs = Number(memory.totalEpochs || 0);
    state.metrics.totalTrainedSamples = Number(memory.totalTrainedSamples || 0);
    state.metrics.lastRunTrainedSamples = Number(memory.lastRunTrainedSamples || 0);
    state.metrics.totalTrainingRunCount = Number(memory.totalTrainingRunCount || 0);
    state.metrics.totalTrainingMs = Number(memory.totalTrainingMs || 0);
    state.metrics.lastLoss = Number.isFinite(memory.lastLoss) ? memory.lastLoss : null;
    state.metrics.lastLocationLoss = Number.isFinite(memory.lastLocationLoss) ? memory.lastLocationLoss : null;
    state.metrics.lastGridLoss = Number.isFinite(memory.lastGridLoss) ? memory.lastGridLoss : null;
    state.metrics.lastScalarLoss = Number.isFinite(memory.lastScalarLoss) ? memory.lastScalarLoss : null;
    state.metrics.lastWaitLoss = Number.isFinite(memory.lastWaitLoss) ? memory.lastWaitLoss : null;
    state.metrics.lastRegionLoss = Number.isFinite(memory.lastRegionLoss) ? memory.lastRegionLoss : null;
    state.metrics.lastLargeEventLoss = Number.isFinite(memory.lastLargeEventLoss) ? memory.lastLargeEventLoss : null;
    state.metrics.currentLossWeights = {
      ...BASE_LOSS_WEIGHTS,
      ...(memory.currentLossWeights || {}),
    };
    state.metrics.regionClassWeights = {
      ...Object.fromEntries(REGION_ORDER.map((region) => [region, 1])),
      ...(memory.regionClassWeights || {}),
    };
    state.metrics.largeEventPositiveRate = Number.isFinite(memory.largeEventPositiveRate) ? memory.largeEventPositiveRate : null;
    state.metrics.largeEventPositiveWeight = Number.isFinite(memory.largeEventPositiveWeight) ? memory.largeEventPositiveWeight : 1;
    state.metrics.missedMatchCount = Number(memory.missedMatchCount || 0);
    state.metrics.lastRegionCombinationAt = memory.lastRegionCombinationAt || null;
    state.regionCombinationPatterns = Array.isArray(memory.regionCombinationPatterns) ? memory.regionCombinationPatterns : [];
    state.metrics.meanDistanceKm = Number.isFinite(memory.meanDistanceKm) ? memory.meanDistanceKm : null;
    state.metrics.medianDistanceKm = Number.isFinite(memory.medianDistanceKm) ? memory.medianDistanceKm : null;
    state.metrics.meanWaitErrorMinutes = Number.isFinite(memory.meanWaitErrorMinutes) ? memory.meanWaitErrorMinutes : null;
    state.metrics.lastTrainingAt = memory.lastTrainingAt || null;
    state.metrics.lastTrainingMs = Number.isFinite(memory.lastTrainingMs) ? memory.lastTrainingMs : null;
    state.metrics.trainingProgress = {
      ...state.metrics.trainingProgress,
      ...(memory.trainingProgress || {}),
      active: false,
      etaMs: null,
      updatedAt: null,
    };
    state.metrics.tableMetricsResetAt = memory.tableMetricsResetAt || null;
    state.metrics.confusionMetricsResetAt = memory.confusionMetricsResetAt || null;
  } catch {
    persistedLearningMs = 0;
    state.metrics.totalLearningMs = currentLearningMs();
    state.metrics.tableMetricsResetAt = null;
    state.metrics.confusionMetricsResetAt = null;
    state.metrics.regionClassWeights = Object.fromEntries(REGION_ORDER.map((region) => [region, 1]));
  }
}

function resetTrainingProgress() {
  state.metrics.trainingProgress = {
    active: false,
    mode: null,
    phase: null,
    phaseLabel: null,
    overallEpoch: 0,
    overallEpochs: 0,
    currentEpoch: 0,
    phaseEpochs: 0,
    currentBatch: 0,
    totalBatches: 0,
    completedUnits: 0,
    totalUnits: 0,
    progressRatio: 0,
    etaMs: null,
    startedAt: null,
    updatedAt: null,
  };
}

function updateTrainingProgress(next) {
  state.metrics.trainingProgress = {
    ...state.metrics.trainingProgress,
    ...next,
    updatedAt: new Date().toISOString(),
  };
}

async function writeJsonAtomic(targetPath, value) {
  const tempPath = `${targetPath}.tmp`;
  const payload = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  await fs.writeFile(tempPath, payload);
  await fs.rename(tempPath, targetPath);
}

async function saveRuntimeState() {
  await writeJsonAtomic(RUNTIME_STATE_FILE, runtimePayload());
}

function saveRuntimeStateSync() {
  try {
    writeFileSync(RUNTIME_STATE_FILE, JSON.stringify(runtimePayload(), null, 2));
  } catch (error) {
    console.error(`Runtime state could not be saved: ${error.message}`);
  }
}

function parseAfadDate(value) {
  return new Date(`${value.replace(" ", "T")}Z`);
}

function pad(n) {
  return String(n).padStart(2, "0");
}

function formatUtcForAfad(date) {
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`;
}

function turkeyDateToAfadUtc(value) {
  return formatUtcForAfad(new Date(`${value.replace(" ", "T")}+03:00`));
}

function normalizeEvent(item) {
  const latitude = Number(item.latitude);
  const longitude = Number(item.longitude);
  const depth = Number(item.depth || 0);
  const magnitude = Number(item.magnitude || 0);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return {
    id: String(item.eventID ?? `${item.date}-${latitude}-${longitude}`),
    date: item.date,
    timestamp: parseAfadDate(item.date).getTime(),
    latitude,
    longitude,
    depth,
    magnitude,
    type: item.type,
    location: item.location || "Bilinmeyen",
    province: item.province || "",
    district: item.district || "",
    country: item.country || "",
  };
}

function sortEvents(events) {
  return events.sort((a, b) => a.timestamp - b.timestamp);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function scale(value, min, max) {
  return clamp((value - min) / (max - min), 0, 1);
}

function unscale(value, min, max) {
  return value * (max - min) + min;
}

function regionBounds(region) {
  return REGION_BOUNDS[region] || REGION_BOUNDS["Diğer"];
}

function scaleLatitudeWithinRegion(latitude, region) {
  const bounds = regionBounds(region);
  return scale(latitude, bounds.latMin, bounds.latMax);
}

function scaleLongitudeWithinRegion(longitude, region) {
  const bounds = regionBounds(region);
  return scale(longitude, bounds.lonMin, bounds.lonMax);
}

function unscaleLatitudeWithinRegion(value, region) {
  const bounds = regionBounds(region);
  return unscale(value, bounds.latMin, bounds.latMax);
}

function unscaleLongitudeWithinRegion(value, region) {
  const bounds = regionBounds(region);
  return unscale(value, bounds.lonMin, bounds.lonMax);
}

function regionLocalCoordinates(event, region = classifyRegion(event)) {
  return {
    latNorm: scaleLatitudeWithinRegion(event.latitude, region),
    lonNorm: scaleLongitudeWithinRegion(event.longitude, region),
  };
}

function gridCellIndexFromLocal(latNorm, lonNorm) {
  const row = clamp(Math.floor(latNorm * GRID_ROWS), 0, GRID_ROWS - 1);
  const col = clamp(Math.floor(lonNorm * GRID_COLS), 0, GRID_COLS - 1);
  return row * GRID_COLS + col;
}

function gridCellIndexForEvent(event, region = classifyRegion(event)) {
  const { latNorm, lonNorm } = regionLocalCoordinates(event, region);
  return gridCellIndexFromLocal(latNorm, lonNorm);
}

function gridCellVector(index) {
  const vector = Array(GRID_CELL_COUNT).fill(0);
  vector[clamp(index, 0, GRID_CELL_COUNT - 1)] = 1;
  return vector;
}

function gridCellFractions(index) {
  const safeIndex = clamp(index, 0, GRID_CELL_COUNT - 1);
  const row = Math.floor(safeIndex / GRID_COLS);
  const col = safeIndex % GRID_COLS;
  return {
    latMinFrac: row / GRID_ROWS,
    lonMinFrac: col / GRID_COLS,
    latSpanFrac: 1 / GRID_ROWS,
    lonSpanFrac: 1 / GRID_COLS,
  };
}

function localOffsetWithinGridCell(event, region = classifyRegion(event)) {
  const { latNorm, lonNorm } = regionLocalCoordinates(event, region);
  const cell = gridCellFractions(gridCellIndexFromLocal(latNorm, lonNorm));
  return [
    clamp((latNorm - cell.latMinFrac) / cell.latSpanFrac, 0, 1),
    clamp((lonNorm - cell.lonMinFrac) / cell.lonSpanFrac, 0, 1),
  ];
}

function scaleWaitMinutes(minutes) {
  const boundedMinutes = clamp(minutes, 0, WAIT_MINUTES_MAX);
  return Math.log1p(boundedMinutes) / Math.log1p(WAIT_MINUTES_MAX);
}

function unscaleWaitMinutes(value) {
  const boundedValue = clamp(value, 0, 1);
  const minutes = Math.expm1(boundedValue * Math.log1p(WAIT_MINUTES_MAX));
  return minutes < MIN_PREDICTED_WAIT_MINUTES ? 0 : minutes;
}

const REGION_GLOBAL_MINS = REGION_ORDER.map((region) => ([
  scale(regionBounds(region).latMin, BOUNDS.latMin, BOUNDS.latMax),
  scale(regionBounds(region).lonMin, BOUNDS.lonMin, BOUNDS.lonMax),
]));

const REGION_GLOBAL_SPANS = REGION_ORDER.map((region) => ([
  Math.max(0.001, scale(regionBounds(region).latMax, BOUNDS.latMin, BOUNDS.latMax) - scale(regionBounds(region).latMin, BOUNDS.latMin, BOUNDS.latMax)),
  Math.max(0.001, scale(regionBounds(region).lonMax, BOUNDS.lonMin, BOUNDS.lonMax) - scale(regionBounds(region).lonMin, BOUNDS.lonMin, BOUNDS.lonMax)),
]));

const GRID_LOCAL_MINS = Array.from({ length: GRID_CELL_COUNT }, (_, index) => {
  const cell = gridCellFractions(index);
  return [cell.latMinFrac, cell.lonMinFrac];
});

const GRID_LOCAL_SPANS = Array.from({ length: GRID_CELL_COUNT }, (_, index) => {
  const cell = gridCellFractions(index);
  return [cell.latSpanFrac, cell.lonSpanFrac];
});

const GLOBAL_LON_SPAN_KM_AT_EQ = (BOUNDS.lonMax - BOUNDS.lonMin) * 111;
const GLOBAL_LAT_SPAN_KM = (BOUNDS.latMax - BOUNDS.latMin) * 111;

const REGION_LON_KM_SPANS = REGION_ORDER.map((region) => {
  const bounds = regionBounds(region);
  const midLat = (bounds.latMin + bounds.latMax) / 2;
  return Math.max(10, (bounds.lonMax - bounds.lonMin) * 111 * Math.max(0.2, Math.cos((midLat * Math.PI) / 180)));
});

function centroidOf(events) {
  if (!events.length) return null;
  return {
    latitude: mean(events.map((item) => item.latitude)),
    longitude: mean(events.map((item) => item.longitude)),
  };
}

function distancesToPoint(events, point) {
  if (!point) return [];
  return events.map((item) => haversineKm(item, point));
}

function regionDominance(events) {
  if (!events.length) return 0;
  const counts = new Map();
  for (const item of events) {
    const region = classifyRegion(item);
    counts.set(region, (counts.get(region) || 0) + 1);
  }
  return Math.max(...counts.values()) / events.length;
}

function zoneSwitchRate(events) {
  if (events.length < 2) return 0;
  let switches = 0;
  for (let i = 1; i < events.length; i += 1) {
    if (classifyRegion(events[i]) !== classifyRegion(events[i - 1])) switches += 1;
  }
  return switches / (events.length - 1);
}

function regionSequence(events) {
  return events.map((item) => classifyRegion(item));
}

function trailingRegionPattern(events, length) {
  if (events.length < length) return [];
  return regionSequence(events.slice(-length));
}

function countPatternOccurrences(regions, pattern, maxStart = regions.length - pattern.length) {
  if (!pattern.length || regions.length < pattern.length) return 0;
  let count = 0;
  for (let start = 0; start <= maxStart; start += 1) {
    let matches = true;
    for (let offset = 0; offset < pattern.length; offset += 1) {
      if (regions[start + offset] !== pattern[offset]) {
        matches = false;
        break;
      }
    }
    if (matches) count += 1;
  }
  return count;
}

function trailingPatternRepeatMetrics(events, length) {
  if (events.length < length + 1) {
    return { repeatRate: 0, seenBefore: 0 };
  }
  const pattern = trailingRegionPattern(events, length);
  if (!pattern.length) {
    return { repeatRate: 0, seenBefore: 0 };
  }
  const regions = regionSequence(events);
  const totalWindows = regions.length - length + 1;
  const priorWindows = totalWindows - 1;
  if (priorWindows <= 0) {
    return { repeatRate: 0, seenBefore: 0 };
  }
  const priorOccurrences = countPatternOccurrences(regions, pattern, priorWindows - 1);
  return {
    repeatRate: clamp(priorOccurrences / priorWindows, 0, 1),
    seenBefore: Number(priorOccurrences > 0),
  };
}

function motifFeatures(events, targetIndex) {
  const timeline = events.slice(Math.max(0, targetIndex - MOTIF_HISTORY_WINDOW), targetIndex);
  const motif4 = trailingPatternRepeatMetrics(timeline, 4);
  const motif5 = trailingPatternRepeatMetrics(timeline, 5);
  const motif6 = trailingPatternRepeatMetrics(timeline, 6);
  return [
    motif4.repeatRate,
    motif5.repeatRate,
    motif6.repeatRate,
    motif4.seenBefore,
    motif5.seenBefore,
    motif6.seenBefore,
  ];
}

function recentInterEventHours(events) {
  if (events.length < 2) return [];
  const deltas = [];
  for (let i = 1; i < events.length; i += 1) {
    deltas.push((events[i].timestamp - events[i - 1].timestamp) / 3_600_000);
  }
  return deltas.filter((value) => Number.isFinite(value) && value >= 0);
}

function normalizedWaitCompression(shortMeanHours, longMeanHours) {
  if (!Number.isFinite(shortMeanHours) || !Number.isFinite(longMeanHours) || longMeanHours <= 0) return 0.5;
  const ratio = shortMeanHours / longMeanHours;
  return clamp(ratio / 2, 0, 1);
}

function hoursSinceLastMagnitude(events, threshold) {
  if (!events.length) return WAIT_HOURS_MAX;
  const latest = events.at(-1);
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (events[i].magnitude >= threshold) {
      return clamp((latest.timestamp - events[i].timestamp) / 3_600_000, 0, WAIT_HOURS_MAX);
    }
  }
  return WAIT_HOURS_MAX;
}

function recentMagnitudeStats(events) {
  if (!events.length) return { meanMagnitude: 0, stdMagnitude: 0, maxMagnitude: 0 };
  const magnitudes = events.map((item) => item.magnitude).filter((value) => Number.isFinite(value));
  if (!magnitudes.length) return { meanMagnitude: 0, stdMagnitude: 0, maxMagnitude: 0 };
  return {
    meanMagnitude: mean(magnitudes),
    stdMagnitude: stddev(magnitudes),
    maxMagnitude: Math.max(...magnitudes),
  };
}

function meanCentroidDistance(events) {
  const centroid = centroidOf(events);
  if (!centroid || !events.length) return 0;
  return mean(distancesToPoint(events, centroid));
}

function transitionSurprise(history, event) {
  if (!history.length) return 0;
  const previousRegion = classifyRegion(history.at(-1));
  const currentRegion = classifyRegion(event);
  let fromCount = 0;
  let transitionCount = 0;
  for (let i = 1; i < history.length; i += 1) {
    const fromRegion = classifyRegion(history[i - 1]);
    const toRegion = classifyRegion(history[i]);
    if (fromRegion !== previousRegion) continue;
    fromCount += 1;
    if (toRegion === currentRegion) transitionCount += 1;
  }
  const smoothedProbability = (transitionCount + 1) / (fromCount + REGION_ORDER.length);
  const maxSurprise = Math.log(Math.max(2, fromCount + REGION_ORDER.length));
  if (!Number.isFinite(maxSurprise) || maxSurprise <= 0) return 0;
  return clamp((-Math.log(smoothedProbability)) / maxSurprise, 0, 1);
}

function transitionProbability(history, sourceRegion, targetRegion) {
  if (!history.length) return 0;
  let sourceCount = 0;
  let transitionCount = 0;
  for (let i = 1; i < history.length; i += 1) {
    const fromRegion = classifyRegion(history[i - 1]);
    const toRegion = classifyRegion(history[i]);
    if (fromRegion !== sourceRegion) continue;
    sourceCount += 1;
    if (toRegion === targetRegion) transitionCount += 1;
  }
  return clamp((transitionCount + 1) / (sourceCount + REGION_ORDER.length), 0, 1);
}

function triggerPersistenceScore(history, sourceRegion, targetRegion) {
  if (history.length < 2) return 0;
  const recent = history.slice(-(TRANSITION_RECENT_WINDOW + 1));
  let opportunities = 0;
  let matches = 0;
  for (let i = 1; i < recent.length; i += 1) {
    const fromRegion = classifyRegion(recent[i - 1]);
    const toRegion = classifyRegion(recent[i]);
    if (fromRegion !== sourceRegion) continue;
    opportunities += 1;
    if (toRegion === targetRegion) matches += 1;
  }
  if (!opportunities) return 0;
  return clamp(matches / opportunities, 0, 1);
}

function activationLift(history, sourceRegion, targetRegion) {
  if (history.length < 2) return 0;
  let sourceCount = 0;
  let transitionCount = 0;
  let targetCount = 0;
  for (let i = 0; i < history.length; i += 1) {
    const region = classifyRegion(history[i]);
    if (region === targetRegion) targetCount += 1;
    if (i === 0) continue;
    const fromRegion = classifyRegion(history[i - 1]);
    const toRegion = region;
    if (fromRegion === sourceRegion) {
      sourceCount += 1;
      if (toRegion === targetRegion) transitionCount += 1;
    }
  }
  const targetBaseline = (targetCount + 1) / (history.length + REGION_ORDER.length);
  const postSourceActivation = (transitionCount + 1) / (sourceCount + REGION_ORDER.length);
  const lift = postSourceActivation / Math.max(targetBaseline, 1e-6);
  return clamp(lift / 3, 0, 1);
}

function sourcePressure(history, sourceRegion) {
  if (!history.length) return 0;
  const recent = history.slice(-PRESSURE_RECENT_WINDOW);
  const sourceEvents = recent.filter((item) => classifyRegion(item) === sourceRegion);
  if (!sourceEvents.length) return 0;
  const frequencyScore = sourceEvents.length / recent.length;
  const energyScore = mean(sourceEvents.map((item) => clamp(Number(item.magnitude) / BOUNDS.magMax, 0, 1)));
  return clamp((frequencyScore * 0.6) + (energyScore * 0.4), 0, 1);
}

function targetSensitivity(history, targetRegion) {
  if (history.length < 2) return 0;
  let externalTransitions = 0;
  let targetActivations = 0;
  const uniqueSources = new Set();
  for (let i = 1; i < history.length; i += 1) {
    const fromRegion = classifyRegion(history[i - 1]);
    const toRegion = classifyRegion(history[i]);
    if (fromRegion === targetRegion) continue;
    externalTransitions += 1;
    if (toRegion === targetRegion) {
      targetActivations += 1;
      uniqueSources.add(fromRegion);
    }
  }
  if (!externalTransitions) return 0;
  const activationRate = targetActivations / externalTransitions;
  const sourceCoverage = uniqueSources.size / Math.max(1, REGION_ORDER.length - 1);
  return clamp((activationRate * 0.7) + (sourceCoverage * 0.3), 0, 1);
}

function eventFeatures(event, previous, history = []) {
  const hour = new Date(event.timestamp).getUTCHours() + 3;
  const deltaHours = previous ? clamp((event.timestamp - previous.timestamp) / 3_600_000, 0, 168) / 168 : 0;
  const distanceFromPrevious = previous ? clamp(haversineKm(previous, event) / 900, 0, 1) : 0;
  const recent3 = history.slice(-3);
  const recent5 = history.slice(-5);
  const knownContext = [...history.slice(-11), event];
  const shortContext = knownContext.slice(-4);
  const mediumContext = knownContext.slice(-8);
  const shortWaits = recentInterEventHours(shortContext);
  const mediumWaits = recentInterEventHours(mediumContext);
  const meanWaitShort = shortWaits.length ? mean(shortWaits) : 0;
  const meanWaitMedium = mediumWaits.length ? mean(mediumWaits) : 0;
  const waitCompression = normalizedWaitCompression(meanWaitShort, meanWaitMedium);
  const sinceLastMag25 = hoursSinceLastMagnitude(knownContext, 2.5);
  const sinceLastMag35 = hoursSinceLastMagnitude(knownContext, 3.5);
  const magnitudeStats = recentMagnitudeStats(mediumContext);
  const clusterSpread8Km = meanCentroidDistance(mediumContext);
  const meanDistanceToRecent3 = recent3.length
    ? clamp(mean(recent3.map((item) => haversineKm(item, event))) / 900, 0, 1)
    : 0;
  const meanDistanceToRecent5 = recent5.length
    ? clamp(mean(recent5.map((item) => haversineKm(item, event))) / 900, 0, 1)
    : 0;
  const centroid3 = centroidOf(recent3);
  const centroid5 = centroidOf(recent5);
  const centroidDistance3 = centroid3 ? clamp(haversineKm(event, centroid3) / 900, 0, 1) : 0;
  const centroidDistance5 = centroid5 ? clamp(haversineKm(event, centroid5) / 900, 0, 1) : 0;
  const sourceRegion = previous ? classifyRegion(previous) : null;
  const targetRegion = classifyRegion(event);
  const sameRegionAsPrevious = sourceRegion ? Number(sourceRegion === targetRegion) : 0;
  const recentZoneDominance = recent5.length ? clamp(regionDominance(recent5), 0, 1) : 0;
  const transitionSurpriseScore = transitionSurprise(history, event);
  const transitionProbabilityScore = sourceRegion ? transitionProbability(history, sourceRegion, targetRegion) : 0;
  const triggerPersistence = sourceRegion ? triggerPersistenceScore(history, sourceRegion, targetRegion) : 0;
  const activationLiftScore = sourceRegion ? activationLift(history, sourceRegion, targetRegion) : 0;
  const sourcePressureScore = sourceRegion ? sourcePressure(history, sourceRegion) : 0;
  const targetSensitivityScore = targetSensitivity(history, targetRegion);
  return [
    scale(event.latitude, BOUNDS.latMin, BOUNDS.latMax),
    scale(event.longitude, BOUNDS.lonMin, BOUNDS.lonMax),
    scale(event.depth, 0, BOUNDS.depthMax),
    scale(event.magnitude, 0, BOUNDS.magMax),
    Math.sin((hour / 24) * Math.PI * 2),
    Math.cos((hour / 24) * Math.PI * 2),
    deltaHours,
    distanceFromPrevious,
    meanDistanceToRecent3,
    meanDistanceToRecent5,
    centroidDistance3,
    centroidDistance5,
    sameRegionAsPrevious,
    recentZoneDominance,
    transitionSurpriseScore,
    transitionProbabilityScore,
    triggerPersistence,
    activationLiftScore,
    sourcePressureScore,
    targetSensitivityScore,
    clamp(meanWaitShort / WAIT_HOURS_MAX, 0, 1),
    clamp(meanWaitMedium / WAIT_HOURS_MAX, 0, 1),
    waitCompression,
    clamp(sinceLastMag25 / WAIT_HOURS_MAX, 0, 1),
    clamp(sinceLastMag35 / WAIT_HOURS_MAX, 0, 1),
    scale(magnitudeStats.meanMagnitude, 0, BOUNDS.magMax),
    clamp(magnitudeStats.stdMagnitude / 2, 0, 1),
    clamp(clusterSpread8Km / 900, 0, 1),
    ...regionVector(targetRegion),
  ];
}

function regionVector(region) {
  const vector = Array(REGION_ORDER.length).fill(0);
  const index = Math.max(0, REGION_ORDER.indexOf(region));
  vector[index] = 1;
  return vector;
}

function computeStaticRegionClassWeights(events) {
  const counts = new Map(REGION_ORDER.map((region) => [region, 0]));
  for (const event of events) {
    const region = classifyRegion(event);
    counts.set(region, (counts.get(region) || 0) + 1);
  }
  const nonZeroCounts = [...counts.values()].filter((count) => count > 0).sort((a, b) => a - b);
  const medianCount = nonZeroCounts.length ? nonZeroCounts[Math.floor(nonZeroCounts.length / 2)] : 1;
  return Object.fromEntries(
    REGION_ORDER.map((region) => {
      const count = counts.get(region) || 0;
      const rawWeight = Math.pow(medianCount / Math.max(1, count), REGION_BALANCE_ALPHA);
      const weight = Number(clamp(rawWeight, REGION_BALANCE_WEIGHT_MIN, REGION_BALANCE_WEIGHT_MAX).toFixed(3));
      return [region, weight];
    }),
  );
}

function predictedRegionFromLocation(match) {
  if (!match?.predicted) return "Diğer";
  return classifyRegion(match.predicted);
}

function computeAdaptiveRegionClassWeights(
  events,
  history = state.comparisonHistory,
  previousWeights = state.metrics.regionClassWeights,
) {
  const staticWeights = computeStaticRegionClassWeights(events);
  const usableHistory = Array.isArray(history)
    ? history.filter((match) => match?.actual && Number.isFinite(match?.distanceKm))
    : [];
  if (!usableHistory.length) return staticWeights;

  const globalMeanDistance = mean(usableHistory.map((match) => match.distanceKm));
  const globalUnder250Rate = usableHistory.filter((match) => match.distanceKm <= 250).length / usableHistory.length;
  const grouped = new Map(REGION_ORDER.map((region) => [region, []]));
  for (const match of usableHistory) {
    const actualRegion = classifyRegion(match.actual);
    if (!grouped.has(actualRegion)) grouped.set(actualRegion, []);
    grouped.get(actualRegion).push(match);
  }

  const rawAdaptiveWeights = Object.fromEntries(
    REGION_ORDER.map((region) => {
      const baseWeight = Number(staticWeights?.[region] ?? 1);
      const matches = grouped.get(region) || [];
      if (matches.length < REGION_PERFORMANCE_MIN_SAMPLES || !Number.isFinite(globalMeanDistance) || globalMeanDistance <= 0) {
        return [region, baseWeight];
      }

      const regionMeanDistance = mean(matches.map((match) => match.distanceKm));
      const regionUnder250Rate = matches.filter((match) => match.distanceKm <= 250).length / matches.length;
      const regionRegionAccuracy = matches.filter((match) => predictedRegionFromLocation(match) === region).length / matches.length;
      const globalRegionAccuracy = usableHistory.filter((match) => predictedRegionFromLocation(match) === classifyRegion(match.actual)).length / usableHistory.length;

      const distanceFactor = clamp(
        regionMeanDistance / globalMeanDistance,
        REGION_PERFORMANCE_DISTANCE_CLAMP_MIN,
        REGION_PERFORMANCE_DISTANCE_CLAMP_MAX,
      );
      const under250Factor = clamp(
        (globalUnder250Rate || 0.0001) / Math.max(regionUnder250Rate, 0.0001),
        REGION_PERFORMANCE_RATE_CLAMP_MIN,
        REGION_PERFORMANCE_RATE_CLAMP_MAX,
      );
      const regionAccuracyFactor = clamp(
        (globalRegionAccuracy || 0.0001) / Math.max(regionRegionAccuracy, 0.0001),
        REGION_PERFORMANCE_RATE_CLAMP_MIN,
        REGION_PERFORMANCE_RATE_CLAMP_MAX,
      );

      const performanceFactor = (distanceFactor * 0.45) + (under250Factor * 0.35) + (regionAccuracyFactor * 0.20);
      const adaptiveWeight = baseWeight * performanceFactor;
      return [region, adaptiveWeight];
    }),
  );

  const averageWeight = mean(Object.values(rawAdaptiveWeights));
  const normalizedWeights = Object.fromEntries(
    REGION_ORDER.map((region) => {
      const normalized = Number(rawAdaptiveWeights[region] ?? 1) / Math.max(averageWeight, 0.0001);
      const clamped = clamp(normalized, REGION_BALANCE_WEIGHT_MIN, REGION_BALANCE_WEIGHT_MAX);
      const previous = Number(previousWeights?.[region] ?? staticWeights?.[region] ?? 1);
      const smoothed = previous + ((clamped - previous) * REGION_ADAPTIVE_SMOOTHING);
      return [region, Number(smoothed.toFixed(3))];
    }),
  );

  return normalizedWeights;
}


function normalizeStoredRegion(region) {
  const legacyMap = {
    KAF: "Kuzey Anadolu Fay Zonu",
    DAF: "Doğu Anadolu Fay Zonu",
    Ege: "Batı Anadolu / Ege Graben Zonu",
    "İç Anadolu": "İç Anadolu Sismik Zonu",
    "IÌ‡c Anadolu": "İç Anadolu Sismik Zonu",
    "Kuzey Anadolu / Karadeniz Kuşağı": "Kuzey Anadolu / Karadeniz Kuşağı",
    "Diğer": "Diğer",
  };
  return legacyMap[region] || region || "Diğer";
}

function normalizePredictionRecord(prediction) {
  if (!prediction) return prediction;
  return {
    ...prediction,
    region: normalizeStoredRegion(prediction.region),
    regionProbabilities: prediction.regionProbabilities
      ? Object.fromEntries(
          Object.entries(prediction.regionProbabilities).map(([key, value]) => [normalizeStoredRegion(key), value]),
        )
      : prediction.regionProbabilities,
  };
}

function normalizeMatchRecord(match) {
  if (!match) return match;
  return {
    ...match,
    predicted: normalizePredictionRecord(match.predicted),
    actual: match.actual,
  };
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, ratio) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1));
  return sorted[index];
}

function confusionCounts(matches, threshold = DEFAULT_CONFIDENCE_THRESHOLD) {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let trueNegative = 0;
  for (const match of matches) {
    const predictedPositive = match.predicted.confidence >= threshold;
    const actualPositive = match.distanceKm <= 250;
    if (predictedPositive && actualPositive) truePositive += 1;
    else if (predictedPositive && !actualPositive) falsePositive += 1;
    else if (!predictedPositive && actualPositive) falseNegative += 1;
    else trueNegative += 1;
  }
  return { truePositive, falsePositive, falseNegative, trueNegative };
}

function confidenceMetricsFromCounts(counts) {
  const { truePositive, falsePositive, falseNegative, trueNegative } = counts;
  const confusionTotal = truePositive + falsePositive + falseNegative + trueNegative;
  const precision = (truePositive + falsePositive) > 0
    ? Number((truePositive / (truePositive + falsePositive)).toFixed(2))
    : null;
  const recall = (truePositive + falseNegative) > 0
    ? Number((truePositive / (truePositive + falseNegative)).toFixed(2))
    : null;
  const specificity = (trueNegative + falsePositive) > 0
    ? Number((trueNegative / (trueNegative + falsePositive)).toFixed(2))
    : null;
  return {
    confusionTotal,
    truePositive,
    falsePositive,
    falseNegative,
    trueNegative,
    precision,
    recall,
    specificity,
  };
}

function computeAdaptiveConfidenceThreshold() {
  const matches = confusionMetricsHistory().filter((match) => (
    Number.isFinite(match.distanceKm) && Number.isFinite(match.predicted?.confidence)
  ));
  if (matches.length < 24) return DEFAULT_CONFIDENCE_THRESHOLD;
  let bestThreshold = DEFAULT_CONFIDENCE_THRESHOLD;
  let bestSpecificity = -Infinity;
  let bestPrecision = -Infinity;
  let bestRecall = -Infinity;
  let foundRecallSafeCandidate = false;
  for (let threshold = MIN_CONFIDENCE_THRESHOLD; threshold <= MAX_CONFIDENCE_THRESHOLD + 1e-9; threshold += 0.01) {
    const roundedThreshold = Number(threshold.toFixed(2));
    const counts = confusionCounts(matches, roundedThreshold);
    const metrics = confidenceMetricsFromCounts(counts);
    const predictedPositiveCount = counts.truePositive + counts.falsePositive;
    if (predictedPositiveCount < 8) continue;
    const precision = metrics.precision ?? 0;
    const recall = metrics.recall ?? 0;
    const specificity = metrics.specificity ?? 0;
    const recallSafe = recall >= MIN_TARGET_RECALL;
    if (recallSafe && !foundRecallSafeCandidate) {
      foundRecallSafeCandidate = true;
      bestSpecificity = -Infinity;
      bestPrecision = -Infinity;
      bestRecall = -Infinity;
    }
    if (foundRecallSafeCandidate && !recallSafe) continue;
    const shouldReplace = specificity > bestSpecificity + 1e-9
      || (Math.abs(specificity - bestSpecificity) < 1e-9 && precision > bestPrecision + 1e-9)
      || (Math.abs(specificity - bestSpecificity) < 1e-9 && Math.abs(precision - bestPrecision) < 1e-9 && recall > bestRecall + 1e-9)
      || (Math.abs(specificity - bestSpecificity) < 1e-9 && Math.abs(precision - bestPrecision) < 1e-9 && Math.abs(recall - bestRecall) < 1e-9 && roundedThreshold > bestThreshold);
    if (shouldReplace) {
      bestRecall = recall;
      bestSpecificity = specificity;
      bestPrecision = precision;
      bestThreshold = roundedThreshold;
    }
  }
  return bestThreshold;
}

function predictionRadiusForCalibration(prediction) {
  if (Number.isFinite(prediction?.predictedRadiusKm)) return prediction.predictedRadiusKm;
  if (Number.isFinite(prediction?.predictedMajorAxisKm)) return prediction.predictedMajorAxisKm;
  return state.metrics.meanDistanceKm || state.metrics.medianDistanceKm || 320;
}

function rawConfidenceForCalibration(prediction) {
  if (Number.isFinite(prediction?.rawConfidence)) return prediction.rawConfidence;
  if (Number.isFinite(prediction?.confidence)) return prediction.confidence;
  if (Number.isFinite(prediction?.regionConfidence)) return prediction.regionConfidence;
  return null;
}

function calibratedConfidenceForPrediction(prediction) {
  const rawConfidence = rawConfidenceForCalibration(prediction);
  if (!Number.isFinite(rawConfidence)) return null;
  const targetRadiusKm = predictionRadiusForCalibration(prediction);
  const targetRegion = prediction?.region || null;
  const matches = state.comparisonHistory
    .filter((match) => Number.isFinite(match.distanceKm) && Number.isFinite(rawConfidenceForCalibration(match.predicted)))
    .map((match) => {
      const matchRawConfidence = rawConfidenceForCalibration(match.predicted);
      const matchRadiusKm = predictionRadiusForCalibration(match.predicted);
      const regionPenalty = targetRegion && match.predicted?.region && targetRegion !== match.predicted.region ? 0.35 : 0;
      const distance = Math.abs(matchRawConfidence - rawConfidence) * 2.0
        + Math.abs(matchRadiusKm - targetRadiusKm) / 700
        + regionPenalty;
      return {
        success: match.distanceKm <= 250 ? 1 : 0,
        distance,
      };
    })
    .sort((a, b) => a.distance - b.distance);
  if (matches.length < 12) return Number(clamp(rawConfidence, 0.08, 0.92).toFixed(3));
  const nearest = matches.slice(0, Math.min(24, matches.length));
  const weightedSuccess = nearest.reduce((sum, item) => sum + (item.success / Math.max(0.05, item.distance + 0.05)), 0);
  const weightedTotal = nearest.reduce((sum, item) => sum + (1 / Math.max(0.05, item.distance + 0.05)), 0);
  const empiricalSuccess = weightedTotal > 0 ? weightedSuccess / weightedTotal : 0.5;
  const support = clamp(nearest.length / 24, 0, 1);
  const empiricalWeight = 0.10 + support * 0.10;
  const rawWeight = 1 - empiricalWeight;
  const blended = rawConfidence * rawWeight + empiricalSuccess * empiricalWeight;
  return Number(clamp(blended, 0.08, 0.92).toFixed(3));
}

function confidenceClassForScore(confidence, threshold = DEFAULT_CONFIDENCE_THRESHOLD) {
  if (!Number.isFinite(confidence)) return null;
  const clampedThreshold = clamp(threshold, MIN_CONFIDENCE_THRESHOLD, MAX_CONFIDENCE_THRESHOLD);
  const highThreshold = clamp(clampedThreshold + HIGH_CONFIDENCE_OFFSET, clampedThreshold + 0.05, 0.92);
  if (confidence >= highThreshold) return "Yüksek";
  if (confidence >= clampedThreshold) return "Orta";
  return "Düşük";
}

function stddev(values) {
  if (values.length < 2) return 0;
  const avg = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)));
}

function summaryFeatures(events, targetIndex) {
  const latest = events[targetIndex - 1];
  return SUMMARY_WINDOWS.flatMap((size) => {
    const start = Math.max(0, targetIndex - size);
    const slice = events.slice(start, targetIndex);
    if (!slice.length || !latest) {
      return Array(SUMMARY_FEATURE_SIZE).fill(0);
    }
    const latitudes = slice.map((event) => event.latitude);
    const longitudes = slice.map((event) => event.longitude);
    const depths = slice.map((event) => event.depth);
    const magnitudes = slice.map((event) => event.magnitude);
    const distancesFromLatest = slice.slice(0, -1).map((event) => haversineKm(event, latest));
    const centroid = centroidOf(slice);
    const distancesToCentroid = distancesToPoint(slice, centroid);
    const interEventDistances = slice.slice(1).map((event, index) => haversineKm(slice[index], event));
    const interEventHours = slice.slice(1).map((event, index) => (event.timestamp - slice[index].timestamp) / 3_600_000);
    const recentSixHoursCount = slice.filter((event) => ((latest.timestamp - event.timestamp) / 3_600_000) <= 6).length;
    const spanHours = Math.max(1, (slice.at(-1).timestamp - slice[0].timestamp) / 3_600_000);
    return [
      scale(mean(latitudes), BOUNDS.latMin, BOUNDS.latMax),
      scale(mean(longitudes), BOUNDS.lonMin, BOUNDS.lonMax),
      clamp(stddev(latitudes) / 4, 0, 1),
      clamp(stddev(longitudes) / 4, 0, 1),
      scale(mean(depths), 0, BOUNDS.depthMax),
      scale(mean(magnitudes), 0, BOUNDS.magMax),
      clamp(slice.length / size, 0, 1),
      clamp(slice.length / spanHours / 8, 0, 1),
      clamp(mean(distancesFromLatest) / 900, 0, 1),
      clamp(mean(interEventDistances) / 900, 0, 1),
      clamp(stddev(interEventDistances) / 500, 0, 1),
      clamp(mean(interEventHours) / 168, 0, 1),
      clamp(stddev(interEventHours) / 72, 0, 1),
      centroid ? clamp(haversineKm(latest, centroid) / 900, 0, 1) : 0,
      distancesToCentroid.length ? clamp(mean(distancesToCentroid) / 900, 0, 1) : 0,
      clamp(zoneSwitchRate(slice), 0, 1),
      clamp(regionDominance(slice), 0, 1),
      clamp(recentSixHoursCount / slice.length, 0, 1),
    ];
  });
}

function hasLargeEventInLookahead(events, targetIndex, threshold = LARGE_EVENT_MAGNITUDE_THRESHOLD, horizonHours = LARGE_EVENT_LOOKAHEAD_HOURS) {
  const referenceEvent = events[targetIndex - 1];
  if (!referenceEvent) return 0;
  const horizonMs = horizonHours * 3_600_000;
  for (let i = targetIndex; i < events.length; i += 1) {
    const deltaMs = events[i].timestamp - referenceEvent.timestamp;
    if (deltaMs > horizonMs) break;
    if (events[i].magnitude >= threshold) return 1;
  }
  return 0;
}

function hasObservedLargeEventLookaheadWindow(events, targetIndex, horizonHours = LARGE_EVENT_LOOKAHEAD_HOURS) {
  const referenceEvent = events[targetIndex - 1];
  const latestEvent = events.at(-1);
  if (!referenceEvent || !latestEvent) return false;
  return (latestEvent.timestamp - referenceEvent.timestamp) >= (horizonHours * 3_600_000);
}

function sequenceFeatures(events, targetIndex) {
  const sequence = [];
  for (let j = targetIndex - LOOKBACK; j < targetIndex; j += 1) {
    sequence.push(eventFeatures(
      events[j],
      events[j - 1],
      events.slice(Math.max(0, j - Math.max(MOTIF_HISTORY_WINDOW, TRIGGER_HISTORY_WINDOW)), j),
    ));
  }
  return sequence;
}

const DATASET_BUILD_PROGRESS_CHUNK = 512;

async function buildDataset(events, { phaseLabel = "Eğitim pencereleri hazırlanıyor", onProgress } = {}) {
  const sequences = [];
  const summaries = [];
  const motifs = [];
  const yLocation = [];
  const yGrid = [];
  const yScalar = [];
  const yWait = [];
  const yRegion = [];
  const yLargeEvent = [];
  const totalCandidates = Math.max(0, events.length - LOOKBACK);
  let processedCandidates = 0;
  for (let i = LOOKBACK; i < events.length; i += 1) {
    processedCandidates += 1;
    if (!hasObservedLargeEventLookaheadWindow(events, i)) continue;
    const targetRegion = classifyRegion(events[i]);
    const regionTarget = regionVector(targetRegion);
    const gridIndex = gridCellIndexForEvent(events[i], targetRegion);
    sequences.push(sequenceFeatures(events, i));
    summaries.push(summaryFeatures(events, i));
    motifs.push(motifFeatures(events, i));
    yLocation.push([
      scale(events[i].latitude, BOUNDS.latMin, BOUNDS.latMax),
      scale(events[i].longitude, BOUNDS.lonMin, BOUNDS.lonMax),
    ]);
    yGrid.push(gridCellVector(gridIndex));
    yScalar.push([
      scale(events[i].depth, 0, BOUNDS.depthMax),
      scale(events[i].magnitude, 0, BOUNDS.magMax),
    ]);
    yWait.push([
      scaleWaitMinutes((events[i].timestamp - events[i - 1].timestamp) / 60_000),
    ]);
    yRegion.push(regionTarget);
    yLargeEvent.push([hasLargeEventInLookahead(events, i)]);
    if (
      onProgress
      && (processedCandidates % DATASET_BUILD_PROGRESS_CHUNK === 0 || processedCandidates === totalCandidates)
    ) {
      await onProgress({
        phase: "preparing",
        phaseLabel,
        processedCandidates,
        totalCandidates,
        sampleCount: sequences.length,
      });
      await new Promise((resolve) => setImmediate(resolve));
    }
  }
  return { sequences, summaries, motifs, yLocation, yGrid, yScalar, yWait, yRegion, yLargeEvent };
}

function currentDatasetSampleCount(events = state.events) {
  let count = 0;
  for (let i = LOOKBACK; i < events.length; i += 1) {
    if (hasObservedLargeEventLookaheadWindow(events, i)) count += 1;
  }
  return count;
}

function recentTrainingSlice(events, limit) {
  if (!Array.isArray(events) || events.length <= limit) return events;
  const contextPadding = LOOKBACK + Math.max(...SUMMARY_WINDOWS) + MOTIF_HISTORY_WINDOW + 8;
  return events.slice(-(limit + contextPadding));
}

function createSpatialLoss() {
  return (yTrue, yPred) => tf.tidy(() => {
    const trueLatNorm = yTrue.slice([0, 0], [-1, 1]);
    const trueLonNorm = yTrue.slice([0, 1], [-1, 1]);
    const predLatNorm = yPred.slice([0, 0], [-1, 1]);
    const predLonNorm = yPred.slice([0, 1], [-1, 1]);
    const trueLat = trueLatNorm.mul(tf.scalar(BOUNDS.latMax - BOUNDS.latMin)).add(tf.scalar(BOUNDS.latMin));
    const predLat = predLatNorm.mul(tf.scalar(BOUNDS.latMax - BOUNDS.latMin)).add(tf.scalar(BOUNDS.latMin));
    const midLatRad = trueLat.add(predLat).div(tf.scalar(2)).mul(tf.scalar(Math.PI / 180));
    const dyKm = predLatNorm.sub(trueLatNorm).mul(tf.scalar(GLOBAL_LAT_SPAN_KM));
    const dxKm = predLonNorm
      .sub(trueLonNorm)
      .mul(tf.scalar(GLOBAL_LON_SPAN_KM_AT_EQ))
      .mul(midLatRad.cos().maximum(tf.scalar(0.2)));
    return dxKm.square().add(dyKm.square()).add(1e-6).sqrt().mean().div(tf.scalar(LOCATION_LOSS_REFERENCE_KM));
  });
}

function createScalarLoss() {
  return (yTrue, yPred) => tf.tidy(() => {
    const trueDepth = yTrue.slice([0, 0], [-1, 1]).mul(BOUNDS.depthMax);
    const predDepth = yPred.slice([0, 0], [-1, 1]).mul(BOUNDS.depthMax);
    const trueMagnitude = yTrue.slice([0, 1], [-1, 1]).mul(BOUNDS.magMax);
    const predMagnitude = yPred.slice([0, 1], [-1, 1]).mul(BOUNDS.magMax);
    const depthTerm = predDepth.sub(trueDepth).abs().div(tf.scalar(DEPTH_LOSS_REFERENCE_KM));
    const magnitudeTerm = predMagnitude.sub(trueMagnitude).abs().div(tf.scalar(MAGNITUDE_LOSS_REFERENCE_ML));
    return depthTerm.add(magnitudeTerm).mean().div(tf.scalar(2));
  });
}

function createWaitLoss() {
  const waitLogScale = Math.log1p(WAIT_MINUTES_MAX);
  return (yTrue, yPred) => tf.tidy(() => {
    const trueMinutes = yTrue.mul(waitLogScale).exp().sub(1);
    const predictedMinutes = yPred.mul(waitLogScale).exp().sub(1);
    return predictedMinutes.sub(trueMinutes).abs().div(tf.scalar(WAIT_LOSS_REFERENCE_MINUTES)).mean();
  });
}

function createLocationMseMetric() {
  return (yTrue, yPred) => tf.tidy(() => {
    const trueLocation = yTrue.slice([0, 0], [-1, 2]);
    const predictedLocation = yPred.slice([0, 0], [-1, 2]);
    return trueLocation.sub(predictedLocation).square().mean();
  });
}

function effectiveLossContributions(weights, losses) {
  return {
    location_head: (weights.location_head ?? 0) * (losses.location_head ?? 0),
    grid_head: (weights.grid_head ?? 0) * (losses.grid_head ?? 0),
    scalar_head: (weights.scalar_head ?? 0) * (losses.scalar_head ?? 0),
    wait_head: (weights.wait_head ?? 0) * (losses.wait_head ?? 0),
    region_head: (weights.region_head ?? 0) * (losses.region_head ?? 0),
    large_event_head: (weights.large_event_head ?? 0) * (losses.large_event_head ?? 0),
  };
}

function computeAdaptiveLossWeights() {
  const losses = {
    location_head: state.metrics.lastLocationLoss,
    grid_head: state.metrics.lastGridLoss,
    scalar_head: state.metrics.lastScalarLoss,
    wait_head: state.metrics.lastWaitLoss,
    region_head: state.metrics.lastRegionLoss,
    large_event_head: state.metrics.lastLargeEventLoss,
  };
  const validLosses = Object.values(losses).filter((value) => Number.isFinite(value) && value > 0);
  if (validLosses.length < Object.keys(losses).length) {
    return { ...BASE_LOSS_WEIGHTS };
  }
  const baseContributions = effectiveLossContributions(BASE_LOSS_WEIGHTS, losses);
  const contributionValues = Object.values(baseContributions).filter((value) => Number.isFinite(value) && value > 0);
  if (!contributionValues.length) {
    return { ...BASE_LOSS_WEIGHTS };
  }
  const targetContribution = mean(contributionValues);
  const rebalanced = Object.fromEntries(
    Object.entries(BASE_LOSS_WEIGHTS).map(([name, baseWeight]) => {
      const contribution = Math.max(1e-6, baseContributions[name] || 1e-6);
      const factor = clamp(
        (targetContribution / contribution) ** LOSS_REBALANCE_ALPHA,
        LOSS_WEIGHT_FACTOR_MIN,
        LOSS_WEIGHT_FACTOR_MAX,
      );
      return [name, baseWeight * factor];
    }),
  );
  const baseSum = Object.values(BASE_LOSS_WEIGHTS).reduce((sum, value) => sum + value, 0);
  const rawSum = Object.values(rebalanced).reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(rawSum) || rawSum <= 0) {
    return { ...BASE_LOSS_WEIGHTS };
  }
  const scaleFactor = baseSum / rawSum;
  return Object.fromEntries(
    Object.entries(rebalanced).map(([name, value]) => [name, Number((value * scaleFactor).toFixed(4))]),
  );
}

function createLargeEventLoss(positiveWeight = 1) {
  const safePositiveWeight = Math.max(1, positiveWeight);
  return (yTrue, yPred) => tf.tidy(() => {
    const epsilon = tf.scalar(1e-7);
    const one = tf.scalar(1);
    const clipped = yPred.clipByValue(1e-7, 1 - 1e-7);
    const positive = yTrue.mul(tf.log(clipped.add(epsilon))).mul(-safePositiveWeight);
    const negative = one.sub(yTrue).mul(tf.log(one.sub(clipped).add(epsilon))).mul(-1);
    return positive.add(negative).mean();
  });
}

function createWeightedRegionLoss(regionClassWeights) {
  const orderedWeights = REGION_ORDER.map((region) => Number(regionClassWeights?.[region] ?? 1));
  return (yTrue, yPred) => tf.tidy(() => {
    const weightTensor = tf.tensor1d(orderedWeights);
    const sampleWeights = yTrue.mul(weightTensor).sum(-1);
    const epsilon = tf.scalar(1e-7);
    const clipped = yPred.clipByValue(1e-7, 1 - 1e-7);
    const crossEntropy = yTrue.mul(tf.log(clipped.add(epsilon))).sum(-1).neg();
    return crossEntropy.mul(sampleWeights).mean();
  });
}

function compileModelWithWeights(targetModel, lossWeights = BASE_LOSS_WEIGHTS, options = {}) {
  const mergedLossWeights = { ...BASE_LOSS_WEIGHTS, ...(lossWeights || {}) };
  const largeEventPositiveWeight = options.largeEventPositiveWeight ?? state.metrics.largeEventPositiveWeight ?? 1;
  const regionClassWeights = options.regionClassWeights ?? state.metrics.regionClassWeights;
  targetModel.compile({
    optimizer: tf.train.adam(0.003),
    loss: {
      location_head: createSpatialLoss(),
      grid_head: "categoricalCrossentropy",
      scalar_head: createScalarLoss(),
      wait_head: createWaitLoss(),
      region_head: createWeightedRegionLoss(regionClassWeights),
      large_event_head: createLargeEventLoss(largeEventPositiveWeight),
    },
    lossWeights: mergedLossWeights,
    metrics: {
      location_head: [createLocationMseMetric()],
      grid_head: ["accuracy"],
      scalar_head: ["mse"],
      wait_head: ["mse"],
      region_head: ["accuracy"],
      large_event_head: ["binaryAccuracy"],
    },
  });
}

function createModel() {
  const sequenceInput = tf.input({ shape: [LOOKBACK, EVENT_FEATURE_SIZE], name: "event_sequence" });
  const summaryInput = tf.input({ shape: [SUMMARY_INPUT_SIZE], name: "long_summary" });
  const motifInput = tf.input({ shape: [MOTIF_FEATURE_SIZE], name: "region_motif" });
  const gru = tf.layers.gru({ units: 48, dropout: 0.08, recurrentDropout: 0.04 }).apply(sequenceInput);
  const summaryDense = tf.layers.dense({ units: 32, activation: "relu" }).apply(summaryInput);
  const merged = tf.layers.concatenate().apply([gru, summaryDense]);
  const hidden = tf.layers.dense({ units: 80, activation: "relu" }).apply(merged);
  const dropped = tf.layers.dropout({ rate: 0.12 }).apply(hidden);
  const compact = tf.layers.dense({ units: 32, activation: "relu" }).apply(dropped);
  const regionContextDense = tf.layers.dense({ units: 10, activation: "relu", name: "region_context_dense" }).apply(compact);
  const motifDense = tf.layers.dense({ units: 20, activation: "relu", name: "region_motif_dense" }).apply(motifInput);
  const regionBranch = tf.layers.concatenate().apply([regionContextDense, motifDense]);
  const regionHidden = tf.layers.dense({ units: 18, activation: "relu", name: "region_hidden" }).apply(regionBranch);
  const regionOutput = tf.layers.dense({ units: REGION_ORDER.length, activation: "softmax", name: "region_head" }).apply(regionHidden);
  const gridHiddenInput = tf.layers.concatenate().apply([compact, regionOutput]);
  const gridHidden = tf.layers.dense({ units: 24, activation: "relu", name: "grid_hidden" }).apply(gridHiddenInput);
  const gridOutput = tf.layers.dense({ units: GRID_CELL_COUNT, activation: "softmax", name: "grid_head" }).apply(gridHidden);
  const localizedRegressionInput = tf.layers.concatenate().apply([compact, regionOutput, gridOutput]);
  const localLocationOutput = tf.layers.dense({ units: 2, activation: "sigmoid", name: "local_location_head" }).apply(localizedRegressionInput);
  const regionMinProjectionLayer = tf.layers.dense({ units: 2, useBias: false, trainable: false, name: "region_min_projection" });
  const regionSpanProjectionLayer = tf.layers.dense({ units: 2, useBias: false, trainable: false, name: "region_span_projection" });
  const gridMinProjectionLayer = tf.layers.dense({ units: 2, useBias: false, trainable: false, name: "grid_min_projection" });
  const gridSpanProjectionLayer = tf.layers.dense({ units: 2, useBias: false, trainable: false, name: "grid_span_projection" });
  const regionMinProjection = regionMinProjectionLayer.apply(regionOutput);
  const regionSpanProjection = regionSpanProjectionLayer.apply(regionOutput);
  const gridMinProjection = gridMinProjectionLayer.apply(gridOutput);
  const gridSpanProjection = gridSpanProjectionLayer.apply(gridOutput);
  const scaledCellOffset = tf.layers.multiply().apply([localLocationOutput, gridSpanProjection]);
  const gridLocationWithinRegion = tf.layers.add().apply([gridMinProjection, scaledCellOffset]);
  const scaledLocalLocation = tf.layers.multiply().apply([gridLocationWithinRegion, regionSpanProjection]);
  const locationOutput = tf.layers.add({ name: "location_head" }).apply([regionMinProjection, scaledLocalLocation]);
  const scalarOutput = tf.layers.dense({ units: 2, activation: "sigmoid", name: "scalar_head" }).apply(compact);
  const waitOutput = tf.layers.dense({ units: 1, activation: "sigmoid", name: "wait_head" }).apply(compact);
  const largeEventHidden = tf.layers.dense({ units: 16, activation: "relu", name: "large_event_hidden" }).apply(compact);
  const largeEventOutput = tf.layers.dense({ units: 1, activation: "sigmoid", name: "large_event_head" }).apply(largeEventHidden);
  const next = tf.model({ inputs: [sequenceInput, summaryInput, motifInput], outputs: [locationOutput, gridOutput, scalarOutput, waitOutput, regionOutput, largeEventOutput] });
  regionMinProjectionLayer.setWeights([tf.tensor2d(REGION_GLOBAL_MINS, [REGION_ORDER.length, 2])]);
  regionSpanProjectionLayer.setWeights([tf.tensor2d(REGION_GLOBAL_SPANS, [REGION_ORDER.length, 2])]);
  gridMinProjectionLayer.setWeights([tf.tensor2d(GRID_LOCAL_MINS, [GRID_CELL_COUNT, 2])]);
  gridSpanProjectionLayer.setWeights([tf.tensor2d(GRID_LOCAL_SPANS, [GRID_CELL_COUNT, 2])]);
  compileModelWithWeights(next, state.metrics.currentLossWeights || BASE_LOSS_WEIGHTS);
  return next;
}

async function loadModel() {
  model = createModel();
  try {
    const memory = JSON.parse(await fs.readFile(MODEL_WEIGHTS_FILE, "utf8"));
    if (memory.version !== MODEL_VERSION) {
      throw new Error(`Unsupported model memory version ${memory.version}`);
    }
    if (memory.summaryInputSize && memory.summaryInputSize !== SUMMARY_INPUT_SIZE) {
      throw new Error(`Unsupported model summary input size ${memory.summaryInputSize}`);
    }
    const tensors = memory.weights.map((weight) => tf.tensor(weight.values, weight.shape));
    model.setWeights(tensors);
    tensors.forEach((tensor) => tensor.dispose());
    state.metrics.modelSavedAt = memory.savedAt || null;
    state.metrics.modelLoadedAt = new Date().toISOString();
    state.metrics.modelMemory = "loaded";
  } catch (error) {
    if (error.code !== "ENOENT") {
      console.warn(`Model memory could not be loaded: ${error.message}`);
    }
    state.metrics.modelMemory = "new";
  }
}

async function saveModelMemory() {
  const weights = await Promise.all(
    model.getWeights().map(async (tensor) => ({
      shape: tensor.shape,
      values: Array.from(await tensor.data()),
    })),
  );
  const savedAt = new Date().toISOString();
  await writeJsonAtomic(MODEL_WEIGHTS_FILE, {
    version: MODEL_VERSION,
    savedAt,
    lookback: LOOKBACK,
    summaryWindows: SUMMARY_WINDOWS,
    eventFeatureSize: EVENT_FEATURE_SIZE,
    motifFeatureSize: MOTIF_FEATURE_SIZE,
    summaryInputSize: SUMMARY_INPUT_SIZE,
    largeEventMagnitudeThreshold: LARGE_EVENT_MAGNITUDE_THRESHOLD,
    largeEventLookaheadHours: LARGE_EVENT_LOOKAHEAD_HOURS,
    weights,
  });
  state.metrics.modelSavedAt = savedAt;
  state.metrics.modelMemory = "saved";
}

function isDisposedModelError(error) {
  return /already disposed/i.test(String(error?.message || ""));
}

async function reloadModelFromMemory() {
  try {
    model?.dispose?.();
  } catch {
    // ignore disposal issues while rebuilding the in-memory model
  }
  model = undefined;
  await loadModel();
}

function ensureTrainingFlagConsistency() {
  if (trainingInProgress && state.metrics.status !== "training") {
    state.metrics.status = "training";
  }
  if (!trainingInProgress && localCatalogTrainingActive && state.metrics.status !== "training") {
    state.metrics.status = "training";
  }
  if (!trainingInProgress && trainingStartPending && state.metrics.status !== "fetching") {
    state.metrics.status = "fetching";
  }
  if (!trainingInProgress && !trainingStartPending && state.metrics.status === "training") {
    state.metrics.status = initialHistoricalTrainingRequired() ? "ready_for_training" : "learning";
  }
  if (!trainingInProgress && !trainingStartPending && state.metrics.status === "fetching") {
    state.metrics.status = initialHistoricalTrainingRequired() ? "ready_for_training" : "learning";
  }
  if (
    !trainingInProgress
    && !trainingStartPending
    && state.metrics.lastAction === "AFAD kontrolü veya eğitim zaten sürüyor."
  ) {
    state.metrics.lastAction = initialHistoricalTrainingRequired()
      ? "İlk eğitim bekleniyor."
      : "Yeni AFAD olayı yok; model canlı izleme durumunda.";
  }
}

function initialHistoricalTrainingRequired() {
  return state.metrics.modelMemory === "new"
    && state.metrics.totalTrainingRunCount === 0
    && state.events.length > LOOKBACK + 2;
}

async function loadEvents() {
  try {
    const raw = JSON.parse(await fs.readFile(EVENTS_FILE, "utf8"));
    state.events = sortEvents(raw.map((event) => ({ ...event, timestamp: parseAfadDate(event.date).getTime() })));
    state.seen = new Set(state.events.map((event) => event.id));
  } catch {
    state.events = [];
    state.seen = new Set();
  }
}

async function saveEvents() {
  await writeJsonAtomic(EVENTS_FILE, state.events);
}

function mainCatalogNeedsRefresh() {
  if (state.events.length <= LOOKBACK + 2) return true;
  const expectedStartTs = new Date(`${START_DATE_TURKEY.replace(" ", "T")}+03:00`).getTime();
  const earliestTs = state.events[0]?.timestamp;
  if (!Number.isFinite(earliestTs)) return true;
  const oneDayMs = 24 * 3_600_000;
  return Math.abs(earliestTs - expectedStartTs) > oneDayMs;
}

async function loadLargeEventTrainingEvents() {
  try {
    const raw = JSON.parse(await fs.readFile(LARGE_EVENT_EVENTS_FILE, "utf8"));
    state.largeEventTrainingEvents = sortEvents(raw.map((event) => ({ ...event, timestamp: parseAfadDate(event.date).getTime() })));
    state.largeEventTrainingSeen = new Set(state.largeEventTrainingEvents.map((event) => event.id));
  } catch {
    state.largeEventTrainingEvents = [];
    state.largeEventTrainingSeen = new Set();
  }
}

async function saveLargeEventTrainingEvents() {
  await writeJsonAtomic(LARGE_EVENT_EVENTS_FILE, state.largeEventTrainingEvents);
}

function largeEventTrainingCatalogNeedsRefresh() {
  if (state.largeEventTrainingEvents.length <= LOOKBACK + 2) return true;
  const expectedStartTs = new Date(`${LARGE_EVENT_START_DATE_TURKEY.replace(" ", "T")}+03:00`).getTime();
  const earliestTs = state.largeEventTrainingEvents[0]?.timestamp;
  if (!Number.isFinite(earliestTs)) return true;
  const oneDayMs = 24 * 3_600_000;
  return Math.abs(earliestTs - expectedStartTs) > oneDayMs;
}

async function loadPredictionState() {
  try {
    const memory = JSON.parse(await fs.readFile(PREDICTION_STATE_FILE, "utf8"));
    state.predictions = (memory.predictions || [])
      .filter((prediction) => Number.isFinite(prediction.waitMinutes))
      .map(normalizePredictionRecord)
      .map(enrichPredictionUncertainty);
    state.previousPrediction = Number.isFinite(memory.previousPrediction?.waitMinutes)
      ? enrichPredictionUncertainty(normalizePredictionRecord(memory.previousPrediction))
      : null;
    if (memory.matchHistorySchemaVersion === MATCH_HISTORY_SCHEMA_VERSION) {
      state.lastMatch = normalizeMatchRecord(memory.lastMatch || null);
      const seenActualIds = new Set();
      const validMatches = (memory.recentMatches || (state.lastMatch ? [state.lastMatch] : []))
        .map(normalizeMatchRecord)
        .filter((match) => {
          if (!match?.actual?.id || !match?.predicted) return false;
          if (match.predicted.basedOnEventId === match.actual.id) return false;
          if (match.predicted.basedOnDate && match.predicted.basedOnDate === match.actual.date) return false;
          if (seenActualIds.has(match.actual.id)) return false;
          seenActualIds.add(match.actual.id);
          return true;
        });
      state.recentMatches = validMatches.slice(0, 2).map(enrichMatchTiming);
      state.comparisonHistory = (memory.comparisonHistory || validMatches)
        .filter((match) => match?.actual?.id && match?.predicted)
        .map(normalizeMatchRecord)
        .map(enrichMatchTiming);
    } else {
      state.lastMatch = null;
      state.recentMatches = [];
      state.comparisonHistory = [];
    }
    state.largeEventSignals = memory.largeEventSignalsSchemaVersion === LARGE_EVENT_SIGNAL_SCHEMA_VERSION
      ? (memory.largeEventSignals || [])
        .filter((item) => item?.basedOnEventId)
        .map((item) => ({
          ...item,
          resolved: Boolean(item.resolved),
          predictedPositive: Boolean(item.predictedPositive),
        }))
      : [];
    state.lastMatch = state.recentMatches[0] || null;
    state.recentMatches = state.recentMatches.slice(0, 1);
    state.lastMatch = state.recentMatches[0] || null;
    state.showMapHistory = Boolean(state.lastMatch);
    if (state.lastMatch?.distanceKm) {
      state.metrics.lastMatchDistanceKm = state.lastMatch.distanceKm;
    }
  } catch {
    state.predictions = [];
    state.previousPrediction = null;
    state.lastMatch = null;
    state.recentMatches = [];
    state.comparisonHistory = [];
    state.largeEventSignals = [];
    state.showMapHistory = true;
  }
}

function actualWaitMinutesFor(event, predicted) {
  const actualEvent = event?.id ? state.events.find((item) => item.id === event.id) : null;
  const actualTimestamp = Number(actualEvent?.timestamp ?? event?.timestamp);
  if (!Number.isFinite(actualTimestamp)) return null;
  const basedOnEventId = predicted?.basedOnEventId;
  if (basedOnEventId) {
    const referenceEvent = state.events.find((item) => item.id === basedOnEventId);
    if (referenceEvent?.timestamp) {
      return Number(((actualTimestamp - referenceEvent.timestamp) / 60_000).toFixed(1));
    }
  }
  const basedAt = predicted?.basedOnDate ? parseAfadDate(predicted.basedOnDate).getTime() : Number.NaN;
  if (Number.isFinite(basedAt)) {
    return Number(((actualTimestamp - basedAt) / 60_000).toFixed(1));
  }
  const index = state.events.findIndex((item) => item.id === event?.id);
  if (index <= 0) return null;
  return Number(((actualTimestamp - state.events[index - 1].timestamp) / 60_000).toFixed(1));
}

function enrichMatchTiming(match) {
  const predictedSnapshot = findStoredPredictionForMatch(match);
  const predictedWaitMinutes = Number.isFinite(match.predicted?.waitMinutes)
    ? match.predicted.waitMinutes
    : Number.isFinite(predictedSnapshot?.waitMinutes)
      ? predictedSnapshot.waitMinutes
      : predictedWaitMinutesFor(match.predicted);
  const predictedConfidence = Number.isFinite(match.predicted?.confidence)
    ? match.predicted.confidence
    : Number.isFinite(predictedSnapshot?.confidence)
      ? predictedSnapshot.confidence
      : Number.isFinite(match.predicted?.regionConfidence)
        ? match.predicted.regionConfidence
        : Number.isFinite(predictedSnapshot?.regionConfidence)
          ? predictedSnapshot.regionConfidence
          : null;
  const actualWaitMinutes = actualWaitMinutesFor(match.actual, {
    ...predictedSnapshot,
    ...match.predicted,
  }) ?? (Number.isFinite(match.actual?.waitMinutes) ? match.actual.waitMinutes : null);
  return {
    ...match,
    predicted: {
      ...predictedSnapshot,
      ...match.predicted,
      waitMinutes: predictedWaitMinutes,
      confidence: predictedConfidence,
      confidenceClass: match.predicted?.confidenceClass || predictedSnapshot?.confidenceClass || null,
      predictedRadiusKm: Number.isFinite(match.predicted?.predictedRadiusKm)
        ? match.predicted.predictedRadiusKm
        : Number.isFinite(predictedSnapshot?.predictedRadiusKm)
          ? predictedSnapshot.predictedRadiusKm
          : null,
      predictedMajorAxisKm: Number.isFinite(match.predicted?.predictedMajorAxisKm)
        ? match.predicted.predictedMajorAxisKm
        : Number.isFinite(predictedSnapshot?.predictedMajorAxisKm)
          ? predictedSnapshot.predictedMajorAxisKm
          : null,
      predictedMinorAxisKm: Number.isFinite(match.predicted?.predictedMinorAxisKm)
        ? match.predicted.predictedMinorAxisKm
        : Number.isFinite(predictedSnapshot?.predictedMinorAxisKm)
          ? predictedSnapshot.predictedMinorAxisKm
          : null,
      predictedAngleDeg: Number.isFinite(match.predicted?.predictedAngleDeg)
        ? match.predicted.predictedAngleDeg
        : Number.isFinite(predictedSnapshot?.predictedAngleDeg)
          ? predictedSnapshot.predictedAngleDeg
          : null,
    },
    actual: {
      ...match.actual,
      ...(actualEventForMatch(match.actual)?.timestamp ? { timestamp: actualEventForMatch(match.actual).timestamp } : {}),
      waitMinutes: actualWaitMinutes,
    },
  };
}

function actualEventForMatch(event) {
  return event?.id ? state.events.find((item) => item.id === event.id) : null;
}

function findStoredPredictionForMatch(match) {
  const candidates = [state.previousPrediction, ...state.predictions].filter(Boolean);
  return candidates.find((prediction) => (
    prediction.basedOnEventId === match?.predicted?.basedOnEventId
    && Math.abs(prediction.latitude - match.predicted.latitude) < 0.000001
    && Math.abs(prediction.longitude - match.predicted.longitude) < 0.000001
  ));
}

function predictedWaitMinutesFor(predicted) {
  if (!predicted?.expectedAfterDate || !predicted?.basedOnDate) return null;
  const expectedAt = new Date(predicted.expectedAfterDate).getTime();
  const basedAt = parseAfadDate(predicted.basedOnDate).getTime();
  if (!Number.isFinite(expectedAt) || !Number.isFinite(basedAt)) return null;
  return Number(((expectedAt - basedAt) / 60_000).toFixed(1));
}

async function savePredictionState() {
  await writeJsonAtomic(PREDICTION_STATE_FILE, {
    savedAt: new Date().toISOString(),
    matchHistorySchemaVersion: MATCH_HISTORY_SCHEMA_VERSION,
    largeEventSignalsSchemaVersion: LARGE_EVENT_SIGNAL_SCHEMA_VERSION,
    predictions: state.predictions,
    previousPrediction: state.previousPrediction,
    lastMatch: state.lastMatch,
    recentMatches: state.recentMatches,
    comparisonHistory: state.comparisonHistory,
    largeEventSignals: state.largeEventSignals,
    showMapHistory: state.showMapHistory,
  });
}

function resetMetricsState() {
  sessionStartedAt = new Date();
  state.metrics.sessionStartedAt = sessionStartedAt.toISOString();
  persistedLearningMs = 0;
  state.metrics.trainedSamples = 0;
  state.metrics.epochs = 0;
  state.metrics.lastLoss = null;
  state.metrics.lastLocationLoss = null;
  state.metrics.lastGridLoss = null;
  state.metrics.lastScalarLoss = null;
  state.metrics.lastWaitLoss = null;
  state.metrics.lastRegionLoss = null;
  state.metrics.lastLargeEventLoss = null;
  state.metrics.currentLossWeights = { ...BASE_LOSS_WEIGHTS };
  state.metrics.regionClassWeights = Object.fromEntries(REGION_ORDER.map((region) => [region, 1]));
  state.metrics.largeEventPositiveRate = null;
  state.metrics.largeEventPositiveWeight = 1;
  state.metrics.missedMatchCount = 0;
  state.metrics.freshEvents = 0;
  state.metrics.fetchCount = 0;
  state.metrics.meanDistanceKm = null;
  state.metrics.medianDistanceKm = null;
  state.metrics.meanWaitErrorMinutes = null;
  state.metrics.lastFetchAt = null;
  state.metrics.lastTrainingAt = null;
  state.metrics.tableMetricsResetAt = null;
  state.metrics.confusionMetricsResetAt = null;
  state.metrics.totalLearningMs = 0;
  state.metrics.totalEpochs = 0;
  state.metrics.totalTrainedSamples = 0;
  state.metrics.lastRunTrainedSamples = 0;
  state.metrics.totalTrainingRunCount = 0;
  state.metrics.trainingRunCount = 0;
  state.metrics.totalTrainingMs = 0;
  state.metrics.lastTrainingMs = null;
  state.metrics.modelLoadedAt = null;
  state.metrics.modelSavedAt = null;
  state.metrics.modelMemory = "new";
  state.metrics.lastRegionCombinationAt = null;
  state.metrics.error = null;
  resetTrainingProgress();
  state.metrics.lastAction = null;
  state.metrics.status = "idle";
  state.metrics.lastMatchDistanceKm = null;
}

async function resetModelAndIndicators() {
  state.predictions = [];
  state.previousPrediction = null;
  state.lastMatch = null;
  state.recentMatches = [];
  state.comparisonHistory = [];
  state.largeEventSignals = [];
  state.showMapHistory = true;
  resetMetricsState();
  try {
    await fs.unlink(MODEL_WEIGHTS_FILE);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  model?.dispose?.();
  model = createModel();
  resetPending = true;
  state.metrics.lastAction = "Model ve göstergeler sıfırlandı. Eğitim için butona bas.";
  await savePredictionState();
  await saveRuntimeState();
  return { trained: false, freshCount: 0, reason: "reset_only" };
}

function tableMetricsHistory(history = state.comparisonHistory) {
  const resetAt = state.metrics.tableMetricsResetAt
    ? new Date(state.metrics.tableMetricsResetAt).getTime()
    : Number.NaN;
  if (!Number.isFinite(resetAt)) return history;
  return history.filter((match) => {
    const matchedAt = match?.matchedAt ? new Date(match.matchedAt).getTime() : Number.NaN;
    return Number.isFinite(matchedAt) && matchedAt >= resetAt;
  });
}

function confusionMetricsHistory(history = state.comparisonHistory) {
  const resetAt = state.metrics.confusionMetricsResetAt
    ? new Date(state.metrics.confusionMetricsResetAt).getTime()
    : Number.NaN;
  if (!Number.isFinite(resetAt)) return history;
  return history.filter((match) => {
    const matchedAt = match?.matchedAt ? new Date(match.matchedAt).getTime() : Number.NaN;
    return Number.isFinite(matchedAt) && matchedAt >= resetAt;
  });
}

async function resetEvaluationHistory() {
  state.metrics.tableMetricsResetAt = new Date().toISOString();
  state.metrics.lastAction = "Değerlendirme tabloları sıfırlandı.";
  await saveRuntimeState();
  return { reset: true };
}

async function resetConfusionHistory() {
  state.metrics.confusionMetricsResetAt = new Date().toISOString();
  state.metrics.lastAction = "TP / FP / FN / TN göstergeleri sıfırlandı.";
  await saveRuntimeState();
  return { reset: true };
}

function currentEndDate() {
  return formatUtcForAfad(new Date());
}

function incrementalFetchStartTurkey() {
  const latestEvent = state.events.at(-1);
  if (!latestEvent?.timestamp) return START_DATE_TURKEY;
  const overlapMs = 24 * 3_600_000;
  const startDate = new Date(Math.max(0, latestEvent.timestamp - overlapMs));
  return `${startDate.getUTCFullYear()}-${pad2(startDate.getUTCMonth() + 1)}-${pad2(startDate.getUTCDate())} ${pad2(startDate.getUTCHours())}:${pad2(startDate.getUTCMinutes())}:${pad2(startDate.getUTCSeconds())}`;
}

async function fetchAfadCatalogFromStart(startDateTurkey, { timeoutMs = FETCH_TIMEOUT_MS } = {}) {
  const url = new URL(AFAD_URL);
  url.searchParams.set("start", turkeyDateToAfadUtc(startDateTurkey));
  url.searchParams.set("end", currentEndDate());
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      headers: { accept: "application/json" },
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`AFAD isteği ${Math.round(timeoutMs / 1000)} sn içinde yanıt vermedi`);
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
  if (!response.ok) throw new Error(`AFAD ${response.status}: ${response.statusText}`);
  const payload = await response.json();
  const items = Array.isArray(payload) ? payload : (payload?.value ?? []);
  return sortEvents(items.map(normalizeEvent).filter(Boolean));
}

async function ensureLargeEventTrainingEvents() {
  if (!largeEventTrainingCatalogNeedsRefresh()) return false;
  const catalog = await fetchAfadCatalogFromStart(LARGE_EVENT_START_DATE_TURKEY, {
    timeoutMs: Math.max(FETCH_TIMEOUT_MS, 180_000),
  });
  state.largeEventTrainingEvents = catalog;
  state.largeEventTrainingSeen = new Set(catalog.map((event) => event.id));
  await saveLargeEventTrainingEvents();
  return true;
}

async function ensureMainCatalogEvents() {
  if (!mainCatalogNeedsRefresh()) return false;
  const catalog = await fetchAfadCatalogFromStart(START_DATE_TURKEY, {
    timeoutMs: Math.max(FETCH_TIMEOUT_MS, 180_000),
  });
  state.events = catalog;
  state.seen = new Set(catalog.map((event) => event.id));
  await saveEvents();
  return true;
}

async function fetchAfadEvents() {
  if (afadCycleSuspended || localCatalogTrainingActive || trainingInProgress || trainingStartPending) {
    state.metrics.status = trainingInProgress ? "training" : (initialHistoricalTrainingRequired() ? "ready_for_training" : "learning");
    state.metrics.error = null;
    state.metrics.lastAction = trainingInProgress
      ? "Yerel eğitim sürerken AFAD canlı kontrolü askıya alındı."
      : "AFAD canlı kontrolü geçici olarak askıya alındı.";
    return [];
  }
  state.metrics.status = "fetching";
  state.metrics.error = null;
  const items = await fetchAfadCatalogFromStart(incrementalFetchStartTurkey());
  const fresh = [];
  for (const item of items) {
    if (!state.seen.has(item.id)) {
      state.seen.add(item.id);
      fresh.push(item);
    }
  }
  if (fresh.length) {
    sortEvents(fresh);
    state.events.push(...fresh);
    state.events = sortEvents(state.events);
    await saveEvents();
    let largeEventCatalogChanged = false;
    for (const event of fresh) {
      if (!state.largeEventTrainingSeen.has(event.id)) {
        state.largeEventTrainingSeen.add(event.id);
        state.largeEventTrainingEvents.push(event);
        largeEventCatalogChanged = true;
      }
    }
    if (largeEventCatalogChanged) {
      state.largeEventTrainingEvents = sortEvents(state.largeEventTrainingEvents);
      await saveLargeEventTrainingEvents();
    }
  }
  state.metrics.fetchCount += 1;
  state.metrics.freshEvents = fresh.length;
  state.metrics.lastFetchAt = new Date().toISOString();
  return fresh;
}

function buildPredictionRecordForReferenceIndex(referenceIndex) {
  if (referenceIndex < LOOKBACK || referenceIndex >= state.events.length) return null;
  const predicted = predictFromIndex(referenceIndex + 1);
  const latest = state.events[referenceIndex];
  const distanceFromLatestKm = haversineKm(predicted, latest);
  const uncertainty = uncertaintyMetrics(predicted.regionConfidence, predicted.region);
  return {
    ...predicted,
    distanceFromLatestKm: Number(distanceFromLatestKm.toFixed(1)),
    waitMinutes: Number(predicted.waitMinutes.toFixed(1)),
    expectedAfterDate: new Date(latest.timestamp + predicted.waitMinutes * 60_000).toISOString(),
    rawConfidence: uncertainty.rawConfidence,
    confidence: uncertainty.confidence,
    confidenceClass: uncertainty.confidenceClass,
    predictedRadiusKm: uncertainty.radiusKm,
    predictedMajorAxisKm: uncertainty.majorAxisKm,
    predictedMinorAxisKm: uncertainty.minorAxisKm,
    predictedAngleDeg: uncertainty.angleDeg,
    basedOnEventId: latest.id,
    basedOnDate: latest.date,
    note: "Bir sonraki AFAD olayı için deneysel konum tahmini",
  };
}

function ensureLargeEventSignalTracked(prediction) {
  if (!prediction?.basedOnEventId) return false;
  const existing = state.largeEventSignals.find((item) => item.basedOnEventId === prediction.basedOnEventId);
  if (existing) return false;
  const basedAt = parseAfadDate(prediction.basedOnDate).getTime();
  const windowEndAt = basedAt + (LARGE_EVENT_LOOKAHEAD_HOURS * 3_600_000);
  state.largeEventSignals.push({
    basedOnEventId: prediction.basedOnEventId,
    basedOnDate: prediction.basedOnDate,
    basedOnTimestamp: basedAt,
    windowEndAt: new Date(windowEndAt).toISOString(),
    risk: Number(prediction.largeEventRisk ?? 0),
    riskClass: prediction.largeEventRiskClass || null,
    predictedPositive: Number(prediction.largeEventRisk ?? 0) >= 0.5,
    threshold: 0.5,
    magnitudeThreshold: LARGE_EVENT_MAGNITUDE_THRESHOLD,
    lookaheadHours: LARGE_EVENT_LOOKAHEAD_HOURS,
    createdAt: new Date().toISOString(),
    resolved: false,
    actualPositive: null,
    resolvedAt: null,
    triggerEventId: null,
  });
  return true;
}

function reconcileLargeEventSignals() {
  if (!state.largeEventSignals.length || !state.events.length) return false;
  const latestObservedTs = state.events.at(-1)?.timestamp;
  if (!Number.isFinite(latestObservedTs)) return false;
  let changed = false;
  for (const signal of state.largeEventSignals) {
    if (signal.resolved) continue;
    const basedOnTs = Number(signal.basedOnTimestamp);
    const windowEndTs = new Date(signal.windowEndAt).getTime();
    const triggerEvent = state.events.find((event) => (
      event.timestamp > basedOnTs
      && event.timestamp <= windowEndTs
      && event.magnitude >= (signal.magnitudeThreshold ?? LARGE_EVENT_MAGNITUDE_THRESHOLD)
    ));
    if (triggerEvent) {
      signal.resolved = true;
      signal.actualPositive = true;
      signal.triggerEventId = triggerEvent.id;
      signal.resolvedAt = new Date().toISOString();
      changed = true;
      continue;
    }
    if (latestObservedTs >= windowEndTs) {
      signal.resolved = true;
      signal.actualPositive = false;
      signal.triggerEventId = null;
      signal.resolvedAt = new Date().toISOString();
      changed = true;
    }
  }
  return changed;
}

function setTrackActivePrediction(track, prediction) {
  track.previousPrediction = track.predictions[0] || track.previousPrediction;
  track.showMapHistory = true;
  track.predictions = prediction ? [prediction] : [];
  if (track === state && prediction) {
    ensureLargeEventSignalTracked(prediction);
  }
}

function setActivePrediction(prediction) {
  setTrackActivePrediction(state, prediction);
}

function setTrackActivePredictionForReferenceIndex(track, referenceIndex) {
  const prediction = buildPredictionRecordForReferenceIndex(referenceIndex);
  setTrackActivePrediction(track, prediction);
  return prediction;
}

function setActivePredictionForReferenceIndex(referenceIndex) {
  return setTrackActivePredictionForReferenceIndex(state, referenceIndex);
}

async function matchPreviousPrediction(freshEvents) {
  if (!freshEvents?.length || !state.predictions.length) return;
  return closeActivePredictionIfPossible();
}

async function closePredictionAgainstActualTrack(track, previousPrediction, nextActualEvent, { persist = false } = {}) {
  if (!previousPrediction || !nextActualEvent) return { status: "missing_prediction_or_actual" };
  if (previousPrediction.basedOnEventId === nextActualEvent.id) {
    if (track === state) {
      state.metrics.missedMatchCount += 1;
      state.metrics.lastAction = `Açık tahmin kapatılamadı: gerçek olay referans olayla aynı görünüyor (${previousPrediction.basedOnEventId}).`;
      await saveRuntimeState();
    }
    if (persist) await savePredictionState();
    return { status: "invalid_self_match", basedOnEventId: previousPrediction.basedOnEventId };
  }
  const existingMatch = track.comparisonHistory.find((match) => (
    match?.actual?.id === nextActualEvent.id
    && match?.predicted?.basedOnEventId === previousPrediction.basedOnEventId
  ));
  if (existingMatch) {
    track.lastMatch = existingMatch;
    track.previousPrediction = previousPrediction;
    track.predictions = [];
    track.showMapHistory = true;
    if (track === state) {
      state.metrics.lastMatchDistanceKm = existingMatch.distanceKm;
    }
    if (persist) await savePredictionState();
    return { status: "already_closed", basedOnEventId: previousPrediction.basedOnEventId, actualId: nextActualEvent.id };
  }
  const match = {
    predicted: {
      latitude: previousPrediction.latitude,
      longitude: previousPrediction.longitude,
      depth: previousPrediction.depth,
      magnitude: previousPrediction.magnitude,
      waitMinutes: previousPrediction.waitMinutes,
      region: previousPrediction.region,
      regionConfidence: previousPrediction.regionConfidence,
      confidence: previousPrediction.confidence,
      confidenceClass: previousPrediction.confidenceClass,
      predictedRadiusKm: previousPrediction.predictedRadiusKm,
      predictedMajorAxisKm: previousPrediction.predictedMajorAxisKm,
      predictedMinorAxisKm: previousPrediction.predictedMinorAxisKm,
      predictedAngleDeg: previousPrediction.predictedAngleDeg,
      expectedAfterDate: previousPrediction.expectedAfterDate,
      basedOnEventId: previousPrediction.basedOnEventId,
      basedOnDate: previousPrediction.basedOnDate,
    },
    actual: {
      ...nextActualEvent,
      waitMinutes: actualWaitMinutesFor(nextActualEvent, previousPrediction),
    },
    distanceKm: Number(haversineKm(previousPrediction, nextActualEvent).toFixed(1)),
    matchedAt: new Date().toISOString(),
  };
  track.lastMatch = match;
  track.recentMatches = [match, ...track.recentMatches].slice(0, 2);
  track.comparisonHistory = [...track.comparisonHistory, match];
  track.previousPrediction = previousPrediction;
  track.predictions = [];
  track.showMapHistory = true;
  if (track === state) {
    state.metrics.lastMatchDistanceKm = track.lastMatch.distanceKm;
  }
  if (persist) await savePredictionState();
  return { status: "closed", basedOnEventId: previousPrediction.basedOnEventId, actualId: nextActualEvent.id };
}

async function closePredictionAgainstActual(previousPrediction, nextActualEvent) {
  return closePredictionAgainstActualTrack(state, previousPrediction, nextActualEvent, { persist: true });
}

async function closeActivePredictionIfPossibleTrack(track, { persist = false } = {}) {
  if (!track.predictions.length) return { status: "no_active" };
  const previousPrediction = track.predictions[0];
  const basedIndex = state.events.findIndex((event) => event.id === previousPrediction.basedOnEventId);
  if (basedIndex < 0) {
    if (track === state) {
      state.metrics.missedMatchCount += 1;
      state.metrics.lastAction = `Açık tahmin kapatılamadı: referans olay bulunamadı (${previousPrediction.basedOnEventId}).`;
      await saveRuntimeState();
    }
    if (persist) await savePredictionState();
    return { status: "missing_reference", basedOnEventId: previousPrediction.basedOnEventId };
  }
  const nextActualEvent = basedIndex >= 0 ? state.events[basedIndex + 1] : null;
  if (!nextActualEvent) return { status: "waiting_next_actual", basedOnEventId: previousPrediction.basedOnEventId };
  return closePredictionAgainstActualTrack(track, previousPrediction, nextActualEvent, { persist });
}

async function closeActivePredictionIfPossible() {
  return closeActivePredictionIfPossibleTrack(state, { persist: true });
}

function haversineKm(a, b) {
  const radius = 6371;
  const toRad = (degree) => (degree * Math.PI) / 180;
  const dLat = toRad(b.latitude - a.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const lat1 = toRad(a.latitude);
  const lat2 = toRad(b.latitude);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(h));
}

function buildTrainingTensors(dataset) {
  return {
    sequenceTensor: tf.tensor3d(dataset.sequences),
    summaryTensor: tf.tensor2d(dataset.summaries),
    motifTensor: tf.tensor2d(dataset.motifs),
    locationTensor: tf.tensor2d(dataset.yLocation),
    gridTensor: tf.tensor2d(dataset.yGrid),
    scalarTensor: tf.tensor2d(dataset.yScalar),
    waitTensor: tf.tensor2d(dataset.yWait),
    regionTensor: tf.tensor2d(dataset.yRegion),
    largeEventTensor: tf.tensor2d(dataset.yLargeEvent),
  };
}

function disposeTrainingTensors(tensors) {
  Object.values(tensors).forEach((tensor) => tensor.dispose());
}

function applyTrainingHistory(history, { epochs, sampleCount, trainingStartedAt, largeEventLossOverride = null }) {
  const loss = history.history.loss.at(-1);
  const locationLoss = history.history.location_head_loss?.at(-1);
  const gridLoss = history.history.grid_head_loss?.at(-1);
  const scalarLoss = history.history.scalar_head_loss?.at(-1);
  const waitLoss = history.history.wait_head_loss?.at(-1);
  const regionLoss = history.history.region_head_loss?.at(-1);
  const largeEventLoss = largeEventLossOverride ?? history.history.large_event_head_loss?.at(-1);
  state.metrics.epochs += epochs;
  state.metrics.trainedSamples += sampleCount;
  state.metrics.totalEpochs += epochs;
  state.metrics.totalTrainedSamples += sampleCount;
  state.metrics.lastRunTrainedSamples = sampleCount;
  state.metrics.lastLoss = Number(loss.toFixed(6));
  state.metrics.lastLocationLoss = Number.isFinite(locationLoss) ? Number(locationLoss.toFixed(6)) : null;
  state.metrics.lastGridLoss = Number.isFinite(gridLoss) ? Number(gridLoss.toFixed(6)) : null;
  state.metrics.lastScalarLoss = Number.isFinite(scalarLoss) ? Number(scalarLoss.toFixed(6)) : null;
  state.metrics.lastWaitLoss = Number.isFinite(waitLoss) ? Number(waitLoss.toFixed(6)) : null;
  state.metrics.lastRegionLoss = Number.isFinite(regionLoss) ? Number(regionLoss.toFixed(6)) : null;
  state.metrics.lastLargeEventLoss = Number.isFinite(largeEventLoss) ? Number(largeEventLoss.toFixed(6)) : null;
  state.metrics.lastTrainingAt = new Date().toISOString();
  state.metrics.lastTrainingMs = Date.now() - trainingStartedAt;
  state.metrics.totalTrainingMs += state.metrics.lastTrainingMs;
  state.metrics.trainingRunCount += 1;
  state.metrics.totalTrainingRunCount += 1;
}

function buildPredictionInputTensors(index) {
  return {
    sequenceTensor: tf.tensor3d([sequenceFeatures(state.events, index)]),
    summaryTensor: tf.tensor2d([summaryFeatures(state.events, index)]),
    motifTensor: tf.tensor2d([motifFeatures(state.events, index)]),
  };
}

function disposePredictionTensors(tensors) {
  Object.values(tensors).forEach((tensor) => tensor.dispose());
}

async function trainModel({ epochs = 6, mode = "full", localOnly = false } = {}) {
  ensureTrainingFlagConsistency();
  if (trainingInProgress) {
    state.metrics.lastAction = "Eğitim zaten sürüyor; yeni istek beklemeye alındı.";
    return;
  }
  if (state.events.length <= LOOKBACK + 2) {
    state.metrics.status = "waiting_for_data";
    await updatePredictions();
    return;
  }
  trainingInProgress = true;
  try {
    const bootstrapLocalTraining = mode === "full"
      && localOnly
      && state.metrics.modelMemory === "new"
      && state.metrics.totalTrainingRunCount === 0;
    const mainEpochs = bootstrapLocalTraining ? INITIAL_BOOTSTRAP_MAIN_EPOCHS : epochs;
    const auxEpochs = bootstrapLocalTraining
      ? INITIAL_BOOTSTRAP_LARGE_EVENT_EPOCHS
      : Math.max(mode === "incremental" ? 1 : 3, Math.ceil(epochs / 2));
    const trainingStartedAt = Date.now();
    console.log("[training] event features:", EVENT_FEATURE_NAMES.join(", "));
    console.log("[training] summary features:", SUMMARY_FEATURE_NAMES.join(", "));
    console.log("[training] motif features:", MOTIF_FEATURE_NAMES.join(", "));
    const trainingActionLabel = mode === "incremental"
      ? "Yeni AFAD olaylarıyla hafif güncelleme yapılıyor."
      : bootstrapLocalTraining
        ? "Mevcut AFAD kataloğuyla ilk güvenli eğitim yapılıyor."
      : "Mevcut AFAD kataloğuyla tam eğitim yapılıyor.";
    const setPreparingState = async (phaseLabel, overrides = {}) => {
      state.metrics.status = "training";
      state.metrics.error = null;
      state.metrics.lastAction = trainingActionLabel;
      updateTrainingProgress({
        active: true,
        mode,
        phase: "preparing",
        phaseLabel,
        overallEpoch: 0,
        overallEpochs: mainEpochs + auxEpochs,
        currentEpoch: 0,
        phaseEpochs: mainEpochs,
        currentBatch: 0,
        totalBatches: state.metrics.trainingProgress?.totalBatches || 0,
        completedUnits: 0,
        totalUnits: state.metrics.trainingProgress?.totalUnits || 0,
        progressRatio: 0,
        etaMs: null,
        startedAt: new Date(trainingStartedAt).toISOString(),
        ...overrides,
      });
      await saveRuntimeState();
      broadcast();
    };
    resetTrainingProgress();
    await setPreparingState(mode === "full" ? "Yerel katalog hazırlanıyor" : "Hafif eğitim hazırlanıyor");
    if (mode === "full" && !localOnly) {
      await setPreparingState("Ana AFAD kataloğu güncelleniyor");
      await ensureMainCatalogEvents();
    }
    if (!localOnly) {
      await setPreparingState("4+ eğitim kataloğu güncelleniyor");
      await ensureLargeEventTrainingEvents();
    }
    await setPreparingState("Bölgesel ağırlıklar hesaplanıyor");
    const fullLargeEventSourceEvents = state.largeEventTrainingEvents.length > LOOKBACK + 2
      ? state.largeEventTrainingEvents
      : state.events;
    const mainTrainingEvents = mode === "incremental"
      ? recentTrainingSlice(state.events, INCREMENTAL_MAIN_EVENT_LIMIT)
      : state.events;
    const largeEventSourceEvents = mode === "incremental"
      ? recentTrainingSlice(fullLargeEventSourceEvents, INCREMENTAL_LARGE_EVENT_LIMIT)
      : fullLargeEventSourceEvents;
    state.metrics.regionClassWeights = computeAdaptiveRegionClassWeights(
      mainTrainingEvents,
      state.comparisonHistory,
      state.metrics.regionClassWeights,
    );
    let datasetProgressLastBroadcastAt = 0;
    const updatePreparingDatasetProgress = ({
      phaseLabel,
      processedCandidates,
      totalCandidates,
      sampleCount,
    }) => {
      const now = Date.now();
      if (now - datasetProgressLastBroadcastAt < 750 && processedCandidates < totalCandidates) return;
      datasetProgressLastBroadcastAt = now;
      const elapsedMs = Date.now() - trainingStartedAt;
      const ratio = totalCandidates > 0 ? processedCandidates / totalCandidates : 0;
      const etaMs = processedCandidates > 0 && ratio > 0
        ? Math.max(0, Math.round((elapsedMs / processedCandidates) * (totalCandidates - processedCandidates)))
        : null;
      updateTrainingProgress({
        active: true,
        mode,
        phase: "preparing",
        phaseLabel: `${phaseLabel} (${processedCandidates}/${totalCandidates}, ${sampleCount} örnek)`,
        overallEpoch: 0,
        overallEpochs: mainEpochs + auxEpochs,
        currentEpoch: 0,
        phaseEpochs: mainEpochs,
        currentBatch: processedCandidates,
        totalBatches: totalCandidates,
        completedUnits: 0,
        totalUnits: 0,
        progressRatio: Number((ratio * 0.2).toFixed(4)),
        etaMs,
        startedAt: new Date(trainingStartedAt).toISOString(),
      });
      saveRuntimeState().catch(() => {});
      broadcast();
    };
    await setPreparingState("Ana eğitim pencereleri hazırlanıyor");
    const mainDataset = await buildDataset(mainTrainingEvents, {
      phaseLabel: "Ana eğitim pencereleri hazırlanıyor",
      onProgress: async (progress) => {
        updatePreparingDatasetProgress(progress);
      },
    });
    await setPreparingState("4+ eğitim pencereleri hazırlanıyor");
    const largeEventDataset = await buildDataset(largeEventSourceEvents, {
      phaseLabel: "4+ eğitim pencereleri hazırlanıyor",
      onProgress: async (progress) => {
        updatePreparingDatasetProgress(progress);
      },
    });
    const positiveCount = largeEventDataset.yLargeEvent.reduce((sum, [value]) => sum + (value >= 0.5 ? 1 : 0), 0);
    const totalCount = largeEventDataset.yLargeEvent.length;
    const negativeCount = Math.max(0, totalCount - positiveCount);
    state.metrics.largeEventPositiveRate = totalCount ? Number((positiveCount / totalCount).toFixed(4)) : null;
    state.metrics.largeEventPositiveWeight = positiveCount > 0
      ? Number(clamp(negativeCount / Math.max(1, positiveCount), 1, 25).toFixed(2))
      : 1;
    const mainBatchSize = Math.min(32, mainDataset.sequences.length);
    const largeEventBatchSize = Math.min(32, largeEventDataset.sequences.length);
    const mainTotalBatches = Math.max(1, Math.ceil(mainDataset.sequences.length / Math.max(1, mainBatchSize)));
    const largeEventTotalBatches = Math.max(1, Math.ceil(largeEventDataset.sequences.length / Math.max(1, largeEventBatchSize)));
    const totalUnits = (mainTotalBatches * mainEpochs) + (largeEventTotalBatches * auxEpochs);
    let progressUnitsCompleted = 0;
    let lastProgressBroadcastAt = 0;
    const maybeBroadcastTrainingProgress = () => {
      const now = Date.now();
      if (now - lastProgressBroadcastAt < 750) return;
      lastProgressBroadcastAt = now;
      saveRuntimeState().catch(() => {});
      broadcast();
    };
    const updateProgressSnapshot = ({
      phase,
      phaseLabel,
      phaseEpochs,
      currentEpoch,
      currentBatch,
      totalBatches,
    }) => {
      const overallEpochOffset = phase === "large_event" ? mainEpochs : 0;
      const completedUnits = Math.min(totalUnits, progressUnitsCompleted);
      const progressRatio = totalUnits > 0 ? completedUnits / totalUnits : 0;
      const elapsedMs = Date.now() - trainingStartedAt;
      const etaMs = completedUnits > 0 && progressRatio > 0
        ? Math.max(0, Math.round((elapsedMs / completedUnits) * (totalUnits - completedUnits)))
        : null;
      updateTrainingProgress({
        active: true,
        mode,
        phase,
        phaseLabel,
        overallEpoch: Math.min(mainEpochs + auxEpochs, overallEpochOffset + currentEpoch),
        overallEpochs: mainEpochs + auxEpochs,
        currentEpoch,
        phaseEpochs,
        currentBatch,
        totalBatches,
        completedUnits,
        totalUnits,
        progressRatio: Number(progressRatio.toFixed(4)),
        etaMs,
        startedAt: new Date(trainingStartedAt).toISOString(),
      });
      maybeBroadcastTrainingProgress();
    };
    updateTrainingProgress({
      active: true,
      mode,
      phase: "preparing",
      phaseLabel: "Tensorler hazırlanıyor",
      overallEpoch: 0,
      overallEpochs: mainEpochs + auxEpochs,
      currentEpoch: 0,
      phaseEpochs: mainEpochs,
      currentBatch: 0,
      totalBatches: mainTotalBatches,
      completedUnits: 0,
      totalUnits,
      progressRatio: 0,
      etaMs: null,
      startedAt: new Date(trainingStartedAt).toISOString(),
    });
    await saveRuntimeState();
    broadcast();
    const runTrainingPass = async (dataset, passEpochs, {
      phase,
      phaseLabel,
      batchSize,
      totalBatches,
      epochOffset,
      checkpointLabel,
    }) => {
      let phaseUnitsCompleted = 0;
      let phaseCurrentEpoch = 1;
      const tensors = buildTrainingTensors(dataset);
      try {
        return await model.fit(
          [tensors.sequenceTensor, tensors.summaryTensor, tensors.motifTensor],
          {
            location_head: tensors.locationTensor,
            grid_head: tensors.gridTensor,
            scalar_head: tensors.scalarTensor,
            wait_head: tensors.waitTensor,
            region_head: tensors.regionTensor,
            large_event_head: tensors.largeEventTensor,
          },
          {
            epochs: passEpochs,
            batchSize,
            shuffle: true,
            verbose: 0,
            yieldEvery: "batch",
            callbacks: {
              onEpochBegin: async (epoch) => {
                phaseCurrentEpoch = epoch + 1;
                updateProgressSnapshot({
                  phase,
                  phaseLabel,
                  phaseEpochs: passEpochs,
                  currentEpoch: epoch + 1,
                  currentBatch: 0,
                  totalBatches,
                });
              },
              onBatchEnd: async (batch) => {
                progressUnitsCompleted += 1;
                phaseUnitsCompleted += 1;
                updateProgressSnapshot({
                  phase,
                  phaseLabel,
                  phaseEpochs: passEpochs,
                  currentEpoch: phaseCurrentEpoch,
                  currentBatch: Math.min(totalBatches, batch + 1),
                  totalBatches,
                });
              },
              onEpochEnd: async (epoch) => {
                updateProgressSnapshot({
                  phase,
                  phaseLabel,
                  phaseEpochs: passEpochs,
                  currentEpoch: epoch + 1,
                  currentBatch: totalBatches,
                  totalBatches,
                });
                if (checkpointLabel) {
                  await updatePredictions();
                  await saveModelMemory();
                  state.metrics.lastAction = `${checkpointLabel} checkpoint kaydedildi (${epoch + 1}/${passEpochs}).`;
                  await saveRuntimeState();
                  await savePredictionState();
                  broadcast();
                }
              },
            },
          },
        );
      } finally {
        disposeTrainingTensors(tensors);
      }
    };
    state.metrics.status = "training";
    state.metrics.error = null;
    state.metrics.lastAction = trainingActionLabel;
    state.metrics.currentLossWeights = computeAdaptiveLossWeights();
    if (!model) await loadModel();
    await setPreparingState("Model derleniyor");
    compileModelWithWeights(model, state.metrics.currentLossWeights, {
      largeEventPositiveWeight: state.metrics.largeEventPositiveWeight,
      regionClassWeights: state.metrics.regionClassWeights,
    });
    let mainHistory;
    let largeEventHistory;
    try {
      mainHistory = await runTrainingPass(mainDataset, mainEpochs, {
        phase: "main",
        phaseLabel: "Ana model eğitimi",
        batchSize: mainBatchSize,
        totalBatches: mainTotalBatches,
        epochOffset: 0,
        checkpointLabel: "Ana model",
      });
    } catch (error) {
      if (!isDisposedModelError(error)) throw error;
      state.metrics.lastAction = "Model belleği tazelendi; yerel eğitim yeniden deneniyor.";
      await reloadModelFromMemory();
      compileModelWithWeights(model, state.metrics.currentLossWeights, {
        largeEventPositiveWeight: state.metrics.largeEventPositiveWeight,
        regionClassWeights: state.metrics.regionClassWeights,
      });
      mainHistory = await runTrainingPass(mainDataset, mainEpochs, {
        phase: "main",
        phaseLabel: "Ana model eğitimi",
        batchSize: mainBatchSize,
        totalBatches: mainTotalBatches,
        epochOffset: 0,
        checkpointLabel: "Ana model",
      });
    }
    await updatePredictions();
    await saveModelMemory();
    state.metrics.lastAction = bootstrapLocalTraining
      ? "Ana model checkpoint kaydedildi; 4+ örüntü eğitimi sürüyor."
      : "Ana model eğitimi tamamlandı; 4+ örüntü eğitimi sürüyor.";
    await saveRuntimeState();
    await savePredictionState();
    broadcast();
    compileModelWithWeights(model, {
      location_head: 0,
      grid_head: 0,
      scalar_head: 0,
      wait_head: 0,
      region_head: 0,
      large_event_head: 1,
    }, {
      largeEventPositiveWeight: state.metrics.largeEventPositiveWeight,
    });
    largeEventHistory = await runTrainingPass(largeEventDataset, auxEpochs, {
      phase: "large_event",
      phaseLabel: "4+ örüntü eğitimi",
      batchSize: largeEventBatchSize,
      totalBatches: largeEventTotalBatches,
      epochOffset: mainEpochs,
      checkpointLabel: "4+ örüntü",
    });
    compileModelWithWeights(model, state.metrics.currentLossWeights, {
      largeEventPositiveWeight: state.metrics.largeEventPositiveWeight,
      regionClassWeights: state.metrics.regionClassWeights,
    });
    applyTrainingHistory(mainHistory, {
      epochs: mainEpochs + auxEpochs,
      sampleCount: mainDataset.sequences.length + largeEventDataset.sequences.length,
      trainingStartedAt,
      largeEventLossOverride: largeEventHistory?.history?.large_event_head_loss?.at(-1) ?? null,
    });
    evaluateModel();
    await updatePredictions();
    await saveModelMemory();
    resetTrainingProgress();
    await saveRuntimeState();
    await savePredictionState();
    state.metrics.status = "learning";
  } finally {
    trainingInProgress = false;
    resetTrainingProgress();
  }
}

async function trainOnlyWhenFresh(freshEvents, { manual = false, forceInitial = false, localOnly = false } = {}) {
  const freshCount = Array.isArray(freshEvents) ? freshEvents.length : Number(freshEvents || 0);
  const orderedFreshEvents = Array.isArray(freshEvents) ? [...freshEvents].sort((a, b) => a.timestamp - b.timestamp) : [];
  if (freshCount > 0) {
    resetPending = false;
    state.metrics.error = null;
    let matchResult = await matchPreviousPrediction(orderedFreshEvents);
    if (state.predictions.length && !["closed", "already_closed", "no_active", undefined].includes(matchResult?.status)) {
      state.metrics.status = "learning";
      state.metrics.lastAction = "Yeni gerçek olay geldi ama açık tahmin kapatılamadığı için yeni tahmin üretilmedi.";
      await saveRuntimeState();
      await savePredictionState();
      return { trained: false, freshCount, reason: "match_blocked", matchResult };
    }
    if (orderedFreshEvents.length > 1) {
      for (let i = 1; i < orderedFreshEvents.length; i += 1) {
        const previousActualEvent = orderedFreshEvents[i - 1];
        const currentActualEvent = orderedFreshEvents[i];
        const referenceIndex = state.events.findIndex((event) => event.id === previousActualEvent.id);
        if (referenceIndex < LOOKBACK) continue;
        const bridgePrediction = buildPredictionRecordForReferenceIndex(referenceIndex);
        if (!bridgePrediction) continue;
        setActivePrediction(bridgePrediction);
        matchResult = await closePredictionAgainstActual(bridgePrediction, currentActualEvent);
        if (!["closed", "already_closed"].includes(matchResult?.status)) {
          state.metrics.status = "learning";
          state.metrics.lastAction = "Ara tahmin zinciri kapanamadığı için yeni tahmin üretilmedi.";
          await saveRuntimeState();
          await savePredictionState();
          return { trained: false, freshCount, reason: "bridge_match_blocked", matchResult };
        }
      }
    }
    if (!state.predictions.length && canGeneratePredictionNow()) {
      await ensureActivePrediction();
      state.metrics.status = "learning";
      state.metrics.error = null;
      state.metrics.lastAction = "Yeni gerçek olay işlendi; mevcut model belleğiyle ara tahmin hazırlandı.";
      await savePredictionState();
      await saveRuntimeState();
    }
    await trainModel({ epochs: manual ? 10 : 2, mode: manual ? "full" : "incremental", localOnly });
    if (!state.predictions.length && canGeneratePredictionNow()) {
      await ensureActivePrediction();
    }
    state.metrics.status = "learning";
    state.metrics.error = null;
    state.metrics.lastAction = manual
      ? `Yeni ${freshCount} AFAD olayı ile tam eğitim yapıldı.`
      : `Yeni ${freshCount} AFAD olayı ile hafif eğitim yapıldı.`;
    return { trained: true, freshCount, reason: "fresh_data" };
  }
  if (state.events.length > LOOKBACK + 2 && (forceInitial || (manual && !state.predictions.length && !resetPending))) {
    resetPending = false;
    state.metrics.error = null;
    await trainModel({ epochs: manual ? 10 : 8, mode: "full", localOnly });
    if (!state.predictions.length && canGeneratePredictionNow()) {
      await ensureActivePrediction();
    }
    state.metrics.status = "learning";
    state.metrics.lastAction = state.predictions.length
      ? "İlk tahmin mevcut veriyle üretildi."
      : "Mevcut katalog eğitimi tamamlandı, tahmin hazırlanıyor.";
    return { trained: true, freshCount, reason: "initial_prediction" };
  }
  if (!freshCount && activePredictionIsStale()) {
    resetPending = false;
    state.metrics.error = null;
    const matchResult = await closeActivePredictionIfPossible();
    if (["closed", "already_closed"].includes(matchResult?.status)) {
      await updatePredictions();
      state.metrics.status = "learning";
      state.metrics.lastAction = "Tahmin son gerçek olaya göre tazelendi.";
    } else {
      state.metrics.status = "learning";
      state.metrics.lastAction = "Açık tahmin kapanamadığı için yeni tahmin üretilmedi.";
    }
    await saveRuntimeState();
    await savePredictionState();
    return { trained: ["closed", "already_closed"].includes(matchResult?.status), freshCount, reason: "prediction_refreshed", matchResult };
  }
  if (!freshCount && !state.predictions.length && canGeneratePredictionNow()) {
    await ensureActivePrediction();
    state.metrics.status = "learning";
    state.metrics.error = null;
    state.metrics.lastAction = manual
      ? "Mevcut model belleğiyle yeni aktif tahmin üretildi."
      : "Yeni AFAD olayı yok; mevcut model belleğiyle aktif tahmin tazelendi.";
    return { trained: false, freshCount, reason: "prediction_created_from_current_model" };
  }
  state.metrics.status = resetPending ? "idle" : "learning";
  state.metrics.error = null;
  state.metrics.lastAction = resetPending
    ? "Model sıfırlandı. Yeni veri gelince ya da butona basınca eğitim başlayacak."
    : manual
      ? "Yeni AFAD olayı yok; model eğitilmedi ve tahmin değiştirilmedi."
      : "Yeni AFAD olayı yok; eğitim atlandı.";
  return { trained: false, freshCount, reason: "no_fresh_data" };
}

function largeEventRiskClass(score) {
  if (!Number.isFinite(score)) return null;
  if (score >= 0.7) return "Yüksek";
  if (score >= 0.4) return "Orta";
  return "Düşük";
}

function predictFromIndex(index) {
  const inputs = buildPredictionInputTensors(index);
  const [locationResult, gridResult, scalarResult, waitResult, regionResult, largeEventResult] = model.predict([
    inputs.sequenceTensor,
    inputs.summaryTensor,
    inputs.motifTensor,
  ]);
  const locationValues = Array.from(locationResult.dataSync());
  const gridValues = Array.from(gridResult.dataSync());
  const scalarValues = Array.from(scalarResult.dataSync());
  const waitValues = Array.from(waitResult.dataSync());
  const regionValues = Array.from(regionResult.dataSync());
  const largeEventValues = Array.from(largeEventResult.dataSync());
  const bestRegionIndex = regionValues.reduce((best, value, current) => (
    value > regionValues[best] ? current : best
  ), 0);
  const bestGridIndex = gridValues.reduce((best, value, current) => (
    value > gridValues[best] ? current : best
  ), 0);
  const predictedRegion = REGION_ORDER[bestRegionIndex];
  disposePredictionTensors(inputs);
  locationResult.dispose();
  gridResult.dispose();
  scalarResult.dispose();
  waitResult.dispose();
  regionResult.dispose();
  largeEventResult.dispose();
  const largeEventRisk = Number(clamp(largeEventValues[0] ?? 0, 0, 1).toFixed(3));
  return {
    latitude: unscale(locationValues[0], BOUNDS.latMin, BOUNDS.latMax),
    longitude: unscale(locationValues[1], BOUNDS.lonMin, BOUNDS.lonMax),
    depth: unscale(scalarValues[0], 0, BOUNDS.depthMax),
    magnitude: unscale(scalarValues[1], 0, BOUNDS.magMax),
    waitMinutes: unscaleWaitMinutes(waitValues[0]),
    region: predictedRegion,
    gridCell: bestGridIndex,
    gridConfidence: gridValues[bestGridIndex],
    regionConfidence: regionValues[bestRegionIndex],
    largeEventRisk,
    largeEventRiskClass: largeEventRiskClass(largeEventRisk),
    gridProbabilities: gridValues.map((value) => Number(value.toFixed(3))),
    regionProbabilities: Object.fromEntries(REGION_ORDER.map((name, idx) => [name, Number(regionValues[idx].toFixed(3))])),
  };
}

function evaluateModel() {
  if (state.events.length <= LOOKBACK + 5) return;
  const start = Math.max(LOOKBACK, state.events.length - 35);
  const distances = [];
  const waitErrors = [];
  const squaredErrors = [];
  for (let i = start; i < state.events.length; i += 1) {
    const predicted = predictFromIndex(i);
    distances.push(haversineKm(predicted, state.events[i]));
    waitErrors.push(Math.abs(predicted.waitMinutes - ((state.events[i].timestamp - state.events[i - 1].timestamp) / 60_000)));
    squaredErrors.push(
      (scale(predicted.latitude, BOUNDS.latMin, BOUNDS.latMax) - scale(state.events[i].latitude, BOUNDS.latMin, BOUNDS.latMax)) ** 2,
      (scale(predicted.longitude, BOUNDS.lonMin, BOUNDS.lonMax) - scale(state.events[i].longitude, BOUNDS.lonMin, BOUNDS.lonMax)) ** 2,
      (scale(predicted.depth, 0, BOUNDS.depthMax) - scale(state.events[i].depth, 0, BOUNDS.depthMax)) ** 2,
      (scale(predicted.magnitude, 0, BOUNDS.magMax) - scale(state.events[i].magnitude, 0, BOUNDS.magMax)) ** 2,
        (scaleWaitMinutes(predicted.waitMinutes) - scaleWaitMinutes((state.events[i].timestamp - state.events[i - 1].timestamp) / 60_000)) ** 2,
      );
  }
  distances.sort((a, b) => a - b);
  const mean = distances.reduce((sum, item) => sum + item, 0) / distances.length;
  state.metrics.meanDistanceKm = Number(mean.toFixed(1));
  state.metrics.medianDistanceKm = Number(distances[Math.floor(distances.length / 2)].toFixed(1));
  state.metrics.meanWaitErrorMinutes = Number((waitErrors.reduce((sum, item) => sum + item, 0) / waitErrors.length).toFixed(1));
  if (state.metrics.lastLoss == null && squaredErrors.length) {
    const meanSquaredError = squaredErrors.reduce((sum, item) => sum + item, 0) / squaredErrors.length;
    state.metrics.lastLoss = Number(meanSquaredError.toFixed(6));
  }
}

function recentPerformanceMetrics(windowSize = 12, history = state.comparisonHistory, lastMatch = state.lastMatch) {
  const recent = history
    .filter((match) => Number.isFinite(match.distanceKm))
    .slice(-windowSize);
  const waitMatches = recent.filter((match) => (
    Number.isFinite(match.predicted?.waitMinutes) && Number.isFinite(match.actual?.waitMinutes)
  ));
  const magnitudeMatches = recent.filter((match) => (
    Number.isFinite(match.predicted?.magnitude) && Number.isFinite(match.actual?.magnitude)
  ));
  const depthMatches = recent.filter((match) => (
    Number.isFinite(match.predicted?.depth) && Number.isFinite(match.actual?.depth)
  ));
  const meanLocationErrorKm = recent.length
    ? Number(mean(recent.map((match) => match.distanceKm)).toFixed(1))
    : null;
  const under250Rate = recent.length
    ? Number(((recent.filter((match) => match.distanceKm <= 250).length / recent.length) * 100).toFixed(0))
    : null;
  const meanWaitErrorMinutes = waitMatches.length
    ? Number(mean(waitMatches.map((match) => Math.abs(match.predicted.waitMinutes - match.actual.waitMinutes))).toFixed(1))
    : null;
  const meanMagnitudeError = magnitudeMatches.length
    ? Number(mean(magnitudeMatches.map((match) => Math.abs(match.predicted.magnitude - match.actual.magnitude))).toFixed(2))
    : null;
  const meanDepthErrorKm = depthMatches.length
    ? Number(mean(depthMatches.map((match) => Math.abs(match.predicted.depth - match.actual.depth))).toFixed(1))
    : null;
  const regionMatches = recent.filter((match) => match.predicted?.region && match.actual);
  const regionAccuracyRate = regionMatches.length
    ? Number(((regionMatches.filter((match) => match.predicted.region === classifyRegion(match.actual)).length / regionMatches.length) * 100).toFixed(0))
    : null;
  const confidenceThreshold = computeAdaptiveConfidenceThreshold();
  const confusion = confusionMetrics("all", confidenceThreshold, history);
  const componentScores = performanceComponentScores({
    locationErrorKm: meanLocationErrorKm,
    under250Rate,
    waitErrorMinutes: meanWaitErrorMinutes,
    magnitudeError: meanMagnitudeError,
    depthErrorKm: meanDepthErrorKm,
    regionAccuracyRate,
    precision: confusion.precision,
    recall: confusion.recall,
    specificity: confusion.specificity,
  });
  const overallScore = componentScores.length
    ? Number(weightedMean(componentScores).toFixed(0))
    : null;
  const latestScore = lastMatch ? singleMatchScore(lastMatch) : null;
  return {
      recentWindow: recent.length,
      meanLocationErrorKm,
    under250Rate,
    meanWaitErrorMinutes,
    meanMagnitudeError,
    meanDepthErrorKm,
    regionAccuracyRate,
    confidenceThreshold,
    confusionWindow: "all",
    confusionTotal: confusion.confusionTotal,
    truePositive: confusion.truePositive,
    falsePositive: confusion.falsePositive,
    falseNegative: confusion.falseNegative,
    trueNegative: confusion.trueNegative,
    precision: confusion.precision,
    recall: confusion.recall,
    specificity: confusion.specificity,
    overallScore,
    latestScore,
  };
}

function largeEventPerformanceMetrics(windowSize = 96, threshold = 0.5) {
  const ready = state.metrics.modelMemory !== "new" && state.metrics.totalTrainingRunCount > 0;
  const sessionStartTs = sessionStartedAt.getTime();
  const sessionSignals = state.largeEventSignals.filter((signal) => {
    const createdAtTs = signal.createdAt ? new Date(signal.createdAt).getTime() : NaN;
    return Number.isFinite(createdAtTs) && createdAtTs >= sessionStartTs;
  });
  const resolvedSignals = sessionSignals
    .filter((signal) => signal.resolved)
    .slice(-windowSize);
  const pendingCount = sessionSignals.filter((signal) => !signal.resolved).length;

  if (!ready) {
    return {
      sampleCount: 0,
      resolvedCount: 0,
      pendingCount,
      positiveCount: 0,
      threshold,
      truePositive: 0,
      falsePositive: 0,
      falseNegative: 0,
      trueNegative: 0,
      precision: null,
      recall: null,
      specificity: null,
      accuracy: null,
      meanRisk: null,
      ready: false,
    };
  }

  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let trueNegative = 0;
  let positiveCount = 0;
  const risks = [];

  for (const signal of resolvedSignals) {
    const predictedPositive = Boolean(signal.predictedPositive);
    const actualPositive = Boolean(signal.actualPositive);
    risks.push(Number(signal.risk ?? 0));
    if (actualPositive) positiveCount += 1;
    if (predictedPositive && actualPositive) truePositive += 1;
    else if (predictedPositive && !actualPositive) falsePositive += 1;
    else if (!predictedPositive && actualPositive) falseNegative += 1;
    else trueNegative += 1;
  }

  const total = resolvedSignals.length;
  const precision = (truePositive + falsePositive) > 0
    ? Number((truePositive / (truePositive + falsePositive)).toFixed(2))
    : null;
  const recall = (truePositive + falseNegative) > 0
    ? Number((truePositive / (truePositive + falseNegative)).toFixed(2))
    : null;
  const specificity = (trueNegative + falsePositive) > 0
    ? Number((trueNegative / (trueNegative + falsePositive)).toFixed(2))
    : null;
  const accuracy = total > 0
    ? Number(((truePositive + trueNegative) / total).toFixed(2))
    : null;

  return {
    sampleCount: total,
    resolvedCount: total,
    pendingCount,
    positiveCount,
    threshold,
    truePositive,
    falsePositive,
    falseNegative,
    trueNegative,
    precision,
    recall,
    specificity,
    accuracy,
    meanRisk: risks.length ? Number(mean(risks).toFixed(2)) : null,
    ready: true,
  };
}

function confusionMetrics(windowSize = "all", threshold = computeAdaptiveConfidenceThreshold(), history = state.comparisonHistory) {
  const filteredHistory = confusionMetricsHistory(history);
  const matchesWindow = windowSize === "all"
    ? filteredHistory
    : filteredHistory
      .filter((match) => Number.isFinite(match.distanceKm))
      .slice(-windowSize);
  const confusionMatches = matchesWindow
    .filter((match) => Number.isFinite(match.distanceKm))
    .map((match) => ({
      ...match,
      predicted: {
        ...match.predicted,
        confidence: calibratedConfidenceForPrediction(match.predicted),
      },
    }))
    .filter((match) => Number.isFinite(match.predicted?.confidence));
  const counts = confusionCounts(confusionMatches, threshold);
  const metrics = confidenceMetricsFromCounts(counts);
  return {
    ...metrics,
    threshold,
  };
}

function regionPerformanceMetrics(windowSize = 60, history = state.comparisonHistory) {
  const filteredHistory = tableMetricsHistory(history);
  const matchesWindow = windowSize === "all"
    ? filteredHistory
    : filteredHistory.slice(-windowSize);
  const groups = new Map(REGION_ORDER.map((name) => [name, []]));
  for (const match of matchesWindow) {
    const region = classifyRegion(match.actual);
    if (!groups.has(region)) groups.set(region, []);
    groups.get(region).push(match);
  }
  return REGION_ORDER.map((region) => {
    const matches = groups.get(region) || [];
    const waitMatches = matches.filter((match) => (
      Number.isFinite(match.predicted?.waitMinutes) && Number.isFinite(match.actual?.waitMinutes)
    ));
    const magnitudeMatches = matches.filter((match) => (
      Number.isFinite(match.predicted?.magnitude) && Number.isFinite(match.actual?.magnitude)
    ));
    const depthMatches = matches.filter((match) => (
      Number.isFinite(match.predicted?.depth) && Number.isFinite(match.actual?.depth)
    ));
    return {
      region,
      count: matches.length,
      meanLocationErrorKm: matches.length ? Number(mean(matches.map((match) => match.distanceKm)).toFixed(1)) : null,
      under250Rate: matches.length
        ? Number(((matches.filter((match) => match.distanceKm <= 250).length / matches.length) * 100).toFixed(0))
        : null,
      meanWaitErrorMinutes: waitMatches.length
        ? Number(mean(waitMatches.map((match) => Math.abs(match.predicted.waitMinutes - match.actual.waitMinutes))).toFixed(1))
        : null,
      meanMagnitudeError: magnitudeMatches.length
        ? Number(mean(magnitudeMatches.map((match) => Math.abs(match.predicted.magnitude - match.actual.magnitude))).toFixed(2))
        : null,
      meanDepthErrorKm: depthMatches.length
        ? Number(mean(depthMatches.map((match) => Math.abs(match.predicted.depth - match.actual.depth))).toFixed(1))
        : null,
    };
  });
}

function regionTransitionMetrics(windowSize = 60, history = state.comparisonHistory) {
  const filteredHistory = tableMetricsHistory(history);
  const matchesWindow = windowSize === "all"
    ? filteredHistory
    : filteredHistory.slice(-windowSize);
  const groups = new Map();
  for (const match of matchesWindow) {
    const fromRegion = classifyRegion(state.events.find((event) => event.id === match.predicted?.basedOnEventId));
    const toRegion = classifyRegion(match.actual);
    const key = `${fromRegion}->${toRegion}`;
    if (!groups.has(key)) {
      groups.set(key, {
        fromRegion,
        toRegion,
        matches: [],
      });
    }
    groups.get(key).matches.push(match);
  }

  return [...groups.values()]
    .map(({ fromRegion, toRegion, matches }) => {
      const waitMatches = matches.filter((match) => (
        Number.isFinite(match.predicted?.waitMinutes) && Number.isFinite(match.actual?.waitMinutes)
      ));
      const magnitudeMatches = matches.filter((match) => (
        Number.isFinite(match.predicted?.magnitude) && Number.isFinite(match.actual?.magnitude)
      ));
      return {
        fromRegion,
        toRegion,
        count: matches.length,
        meanLocationErrorKm: matches.length ? Number(mean(matches.map((match) => match.distanceKm)).toFixed(1)) : null,
        meanWaitErrorMinutes: waitMatches.length
          ? Number(mean(waitMatches.map((match) => Math.abs(match.predicted.waitMinutes - match.actual.waitMinutes))).toFixed(1))
          : null,
        meanMagnitudeError: magnitudeMatches.length
          ? Number(mean(magnitudeMatches.map((match) => Math.abs(match.predicted.magnitude - match.actual.magnitude))).toFixed(2))
          : null,
      };
    })
    .sort((a, b) => b.count - a.count || a.meanLocationErrorKm - b.meanLocationErrorKm)
    .slice(0, 10);
}

const MAGNITUDE_BANDS = [
  { key: "lt1_5", label: "ML < 1.5", min: -Infinity, max: 1.5 },
  { key: "1_5_2_5", label: "ML 1.5 - 2.5", min: 1.5, max: 2.5 },
  { key: "2_5_3_5", label: "ML 2.5 - 3.5", min: 2.5, max: 3.5 },
  { key: "gte3_5", label: "ML 3.5+", min: 3.5, max: Infinity },
];

const DEPTH_BANDS = [
  { key: "0_5", label: "0 - 5 km", min: 0, max: 5 },
  { key: "5_10", label: "5 - 10 km", min: 5, max: 10 },
  { key: "10_20", label: "10 - 20 km", min: 10, max: 20 },
  { key: "gte20", label: "20+ km", min: 20, max: Infinity },
];

function matchesForWindow(windowSize = 60, history = state.comparisonHistory) {
  const filteredHistory = tableMetricsHistory(history);
  return windowSize === "all"
    ? filteredHistory
    : filteredHistory.slice(-windowSize);
}

function inBand(value, band) {
  if (!Number.isFinite(value)) return false;
  return value >= band.min && value < band.max;
}

function aggregateMatchMetrics(matches) {
  const waitMatches = matches.filter((match) => (
    Number.isFinite(match.predicted?.waitMinutes) && Number.isFinite(match.actual?.waitMinutes)
  ));
  const regionMatches = matches.filter((match) => match.predicted?.region && match.actual);
  return {
    count: matches.length,
    meanLocationErrorKm: matches.length ? Number(mean(matches.map((match) => match.distanceKm)).toFixed(1)) : null,
    under250Rate: matches.length
      ? Number(((matches.filter((match) => match.distanceKm <= 250).length / matches.length) * 100).toFixed(0))
      : null,
    meanWaitErrorMinutes: waitMatches.length
      ? Number(mean(waitMatches.map((match) => Math.abs(match.predicted.waitMinutes - match.actual.waitMinutes))).toFixed(1))
      : null,
    regionAccuracyRate: regionMatches.length
      ? Number(((regionMatches.filter((match) => match.predicted.region === classifyRegion(match.actual)).length / regionMatches.length) * 100).toFixed(0))
      : null,
  };
}

function magnitudePerformanceMetrics(windowSize = 60, history = state.comparisonHistory) {
  const matchesWindow = matchesForWindow(windowSize, history);
  return MAGNITUDE_BANDS.map((band) => {
    const matches = matchesWindow.filter((match) => inBand(Number(match.actual?.magnitude), band));
    return {
      band: band.label,
      ...aggregateMatchMetrics(matches),
    };
  });
}

function depthPerformanceMetrics(windowSize = 60, history = state.comparisonHistory) {
  const matchesWindow = matchesForWindow(windowSize, history);
  return DEPTH_BANDS.map((band) => {
    const matches = matchesWindow.filter((match) => inBand(Number(match.actual?.depth), band));
    return {
      band: band.label,
      ...aggregateMatchMetrics(matches),
    };
  });
}

function regionCombinationMetrics(lengths = [4, 5, 6, 7, 8, 9, 10, 11, 12], maxPerLength = 6, minCount = 2) {
  const regions = state.events.map((event) => classifyRegion(event));
  return lengths.map((length) => {
    if (regions.length < length) {
      return { length, totalWindows: 0, repeatedPatterns: 0, patterns: [] };
    }
    const groups = new Map();
    for (let start = 0; start <= regions.length - length; start += 1) {
      const sequence = regions.slice(start, start + length);
      const key = sequence.join(" -> ");
      if (!groups.has(key)) {
        groups.set(key, {
          sequence,
          count: 0,
          firstSeenAt: state.events[start]?.timestamp ?? null,
          lastSeenAt: state.events[start + length - 1]?.timestamp ?? null,
        });
      }
      const item = groups.get(key);
      item.count += 1;
      item.lastSeenAt = state.events[start + length - 1]?.timestamp ?? item.lastSeenAt;
    }

    const patterns = [...groups.values()]
      .filter((item) => item.count >= minCount)
      .sort((a, b) => (
        b.count - a.count
        || (b.lastSeenAt ?? 0) - (a.lastSeenAt ?? 0)
        || a.sequence.join(" -> ").localeCompare(b.sequence.join(" -> "), "tr")
      ))
      .slice(0, maxPerLength)
      .map((item) => ({
        sequence: item.sequence,
        count: item.count,
        firstSeenLabel: Number.isFinite(item.firstSeenAt) ? formatTurkeyClock(item.firstSeenAt) : null,
        lastSeenLabel: Number.isFinite(item.lastSeenAt) ? formatTurkeyClock(item.lastSeenAt) : null,
      }));

    return {
      length,
      totalWindows: regions.length - length + 1,
      repeatedPatterns: patterns.length,
      patterns,
    };
  });
}

async function refreshRegionCombinationPatterns() {
  state.regionCombinationPatterns = regionCombinationMetrics();
  state.metrics.lastRegionCombinationAt = new Date().toISOString();
  await saveRuntimeState();
  return state.regionCombinationPatterns;
}

function performanceComponentScores({
  locationErrorKm,
  under250Rate,
  waitErrorMinutes,
  magnitudeError,
  depthErrorKm,
  regionAccuracyRate = null,
  precision = null,
  recall = null,
  specificity = null,
}) {
  return [
    weightedScore("location", locationErrorKm == null ? null : Math.max(0, 100 - (locationErrorKm / 700) * 100), 0.28),
    weightedScore("under250", under250Rate, 0.18),
    weightedScore("wait", waitErrorMinutes == null ? null : Math.max(0, 100 - (waitErrorMinutes / 180) * 100), 0.10),
    weightedScore("magnitude", magnitudeError == null ? null : Math.max(0, 100 - (magnitudeError / 1.5) * 100), 0.07),
    weightedScore("depth", depthErrorKm == null ? null : Math.max(0, 100 - (depthErrorKm / 25) * 100), 0.07),
    weightedScore("region", regionAccuracyRate, 0.14),
    weightedScore("precision", Number.isFinite(precision) ? precision * 100 : null, 0.08),
    weightedScore("recall", Number.isFinite(recall) ? recall * 100 : null, 0.04),
    weightedScore("specificity", Number.isFinite(specificity) ? specificity * 100 : null, 0.04),
  ].filter((item) => Number.isFinite(item?.score));
}

function weightedScore(name, score, weight) {
  return Number.isFinite(score) ? { name, score, weight } : null;
}

function weightedMean(items) {
  const valid = items.filter((item) => Number.isFinite(item?.score) && Number.isFinite(item?.weight) && item.weight > 0);
  if (!valid.length) return null;
  const totalWeight = valid.reduce((sum, item) => sum + item.weight, 0);
  if (!totalWeight) return null;
  return valid.reduce((sum, item) => sum + item.score * item.weight, 0) / totalWeight;
}

function singleMatchScore(match) {
  const waitErrorMinutes = Number.isFinite(match.predicted?.waitMinutes) && Number.isFinite(match.actual?.waitMinutes)
    ? Math.abs(match.predicted.waitMinutes - match.actual.waitMinutes)
    : null;
  const magnitudeError = Number.isFinite(match.predicted?.magnitude) && Number.isFinite(match.actual?.magnitude)
    ? Math.abs(match.predicted.magnitude - match.actual.magnitude)
    : null;
  const depthErrorKm = Number.isFinite(match.predicted?.depth) && Number.isFinite(match.actual?.depth)
    ? Math.abs(match.predicted.depth - match.actual.depth)
    : null;
  const scores = performanceComponentScores({
    locationErrorKm: match.distanceKm,
    under250Rate: match.distanceKm <= 250 ? 100 : 0,
    waitErrorMinutes,
    magnitudeError,
    depthErrorKm,
  });
  return scores.length ? Number(weightedMean(scores).toFixed(0)) : null;
}

function signedErrorComponentsKm(match) {
  const predicted = match?.predicted;
  const actual = match?.actual;
  if (!predicted || !actual) return null;
  const midLat = ((Number(predicted.latitude) || 0) + (Number(actual.latitude) || 0)) / 2;
  const dxKm = ((Number(actual.longitude) || 0) - (Number(predicted.longitude) || 0)) * 111 * Math.cos((midLat * Math.PI) / 180);
  const dyKm = ((Number(actual.latitude) || 0) - (Number(predicted.latitude) || 0)) * 111;
  if (!Number.isFinite(dxKm) || !Number.isFinite(dyKm)) return null;
  return { dxKm, dyKm };
}

function uncertaintyMetrics(regionConfidence = 0.5, predictedRegion = null) {
  const recentMatchesAll = state.comparisonHistory.slice(-40);
  const regionScoped = predictedRegion
    ? recentMatchesAll.filter((match) => (
        match?.predicted?.region === predictedRegion || classifyRegion(match?.actual) === predictedRegion
      ))
    : [];
  const scopedMatches = regionScoped.length >= 4 ? regionScoped : recentMatchesAll;
  const recentDistances = scopedMatches
    .map((match) => match?.distanceKm)
    .filter((value) => Number.isFinite(value) && value > 0);
  const fallbackRadius = state.metrics.meanDistanceKm || state.metrics.medianDistanceKm || 320;
  const estimatedRadiusKm = recentDistances.length >= 4
    ? percentile(recentDistances, 0.75)
    : fallbackRadius;
  const radiusKm = Math.round(clamp(estimatedRadiusKm, 80, 700));
  const components = scopedMatches
    .map(signedErrorComponentsKm)
    .filter(Boolean);
  let majorAxisKm = Math.round(clamp(radiusKm * 0.9, 70, 700));
  let minorAxisKm = Math.round(clamp(radiusKm * 0.58, 45, 500));
  let angleDeg = 0;
  if (components.length >= 4) {
    const xs = components.map((item) => item.dxKm);
    const ys = components.map((item) => item.dyKm);
    const meanX = mean(xs);
    const meanY = mean(ys);
    const varX = mean(xs.map((value) => (value - meanX) ** 2));
    const varY = mean(ys.map((value) => (value - meanY) ** 2));
    const covXY = mean(xs.map((value, index) => (value - meanX) * (ys[index] - meanY)));
    const trace = varX + varY;
    const root = Math.sqrt(Math.max(0, ((varX - varY) / 2) ** 2 + covXY ** 2));
    const eigenMajor = Math.max(0, trace / 2 + root);
    const eigenMinor = Math.max(0, trace / 2 - root);
    const principalAngle = 0.5 * Math.atan2(2 * covXY, varX - varY);
    majorAxisKm = Math.round(clamp(Math.max(radiusKm * 0.72, Math.sqrt(eigenMajor) * 1.9), 70, 700));
    minorAxisKm = Math.round(clamp(Math.max(radiusKm * 0.4, Math.sqrt(eigenMinor) * 1.9), 40, majorAxisKm));
    angleDeg = Number((principalAngle * 180 / Math.PI).toFixed(1));
  }
  const radiusScore = 1 - clamp((radiusKm - 80) / 620, 0, 1);
  const regionPrecision = regionScoped.length >= 6
    ? regionScoped.filter((match) => match.distanceKm <= 250).length / regionScoped.length
    : null;
  const calibrationScore = Number.isFinite(regionPrecision) ? clamp(regionPrecision, 0, 1) : 0.5;
  const rawConfidence = clamp(
    regionConfidence * 0.45 + radiusScore * 0.35 + calibrationScore * 0.20,
    0.1,
    0.92,
  );
  const calibratedConfidence = calibratedConfidenceForPrediction({
    region: predictedRegion,
    regionConfidence,
    rawConfidence,
    predictedRadiusKm: radiusKm,
    predictedMajorAxisKm: majorAxisKm,
    predictedMinorAxisKm: minorAxisKm,
  });
  const confidenceThreshold = computeAdaptiveConfidenceThreshold();
  const effectiveConfidence = Number.isFinite(calibratedConfidence) ? calibratedConfidence : rawConfidence;
  const confidenceClass = confidenceClassForScore(effectiveConfidence, confidenceThreshold);
  return {
    radiusKm,
    majorAxisKm,
    minorAxisKm,
    angleDeg,
    rawConfidence: Number(rawConfidence.toFixed(3)),
    confidence: Number(effectiveConfidence.toFixed(3)),
    confidenceClass,
  };
}

function toTurkeyHourStart(timestamp) {
  const shifted = timestamp + (3 * 60 * 60 * 1000);
  return Math.floor(shifted / 3_600_000) * 3_600_000 - (3 * 60 * 60 * 1000);
}

function toTurkeyWindowStart(timestamp, windowHours = 12) {
  const shifted = timestamp + (3 * 60 * 60 * 1000);
  const windowMs = windowHours * 3_600_000;
  return Math.floor(shifted / windowMs) * windowMs - (3 * 60 * 60 * 1000);
}

function formatTurkeyWindowLabel(timestamp, windowHours = 12) {
  const local = new Date(timestamp + (3 * 60 * 60 * 1000));
  const endLocal = new Date(local.getTime() + windowHours * 3_600_000);
  const day = String(local.getUTCDate()).padStart(2, "0");
  const month = String(local.getUTCMonth() + 1).padStart(2, "0");
  const hour = String(local.getUTCHours()).padStart(2, "0");
  const endHour = String(endLocal.getUTCHours()).padStart(2, "0");
  return `${day}.${month} ${hour}:00–${endHour}:00`;
}

function formatTurkeyClock(timestamp) {
  const local = new Date(timestamp + (3 * 60 * 60 * 1000));
  const day = String(local.getUTCDate()).padStart(2, "0");
  const month = String(local.getUTCMonth() + 1).padStart(2, "0");
  const hour = String(local.getUTCHours()).padStart(2, "0");
  const minute = String(local.getUTCMinutes()).padStart(2, "0");
  return `${day}.${month} ${hour}:${minute}`;
}

function pickVariant(seed, variants) {
  if (!variants.length) return "";
  const index = Math.abs(Math.floor(seed)) % variants.length;
  return variants[index];
}

function summarizeSpaceTimeBucket(events, previousBucket = null) {
  const centroid = centroidOf(events);
  const spreadDistances = distancesToPoint(events, centroid);
  const spreadKm = spreadDistances.length ? mean(spreadDistances) : 0;
  const regions = events.map((event) => classifyRegion(event));
  const regionCounts = new Map();
  for (const region of regions) {
    regionCounts.set(region, (regionCounts.get(region) || 0) + 1);
  }
  const rankedRegions = [...regionCounts.entries()].sort((a, b) => b[1] - a[1]);
  const dominantRegion = rankedRegions[0]?.[0] || "Diğer";
  const dominantCount = regionCounts.get(dominantRegion) || 0;
  const dominantRatio = dominantCount / Math.max(1, events.length);
  const secondaryRegion = rankedRegions[1]?.[0] || null;
  const secondaryRatio = secondaryRegion ? (regionCounts.get(secondaryRegion) || 0) / Math.max(1, events.length) : 0;
  const uniqueRegionCount = regionCounts.size;
  const switchRate = zoneSwitchRate(events);
  const previousCentroid = previousBucket?.centroid || null;
  const centroidShiftKm = previousCentroid && centroid ? haversineKm(previousCentroid, centroid) : null;
  const previousSpreadKm = Number.isFinite(previousBucket?.spreadKm) ? previousBucket.spreadKm : null;
  const countDelta = previousBucket ? events.length - previousBucket.count : null;
  const tailCount = Math.max(3, Math.ceil(events.length / 3));
  const tailEvents = events.slice(-tailCount);
  const tailRegions = tailEvents.map((event) => classifyRegion(event));
  const tailCentroid = centroidOf(tailEvents);
  const tailSpreadKm = tailEvents.length ? mean(distancesToPoint(tailEvents, tailCentroid)) : spreadKm;
  const tailRegionCounts = new Map();
  for (const region of tailRegions) {
    tailRegionCounts.set(region, (tailRegionCounts.get(region) || 0) + 1);
  }
  const rankedTailRegions = [...tailRegionCounts.entries()].sort((a, b) => b[1] - a[1]);
  const tailDominantRegion = rankedTailRegions[0]?.[0] || dominantRegion;
  const tailDominantRatio = (tailRegionCounts.get(tailDominantRegion) || 0) / Math.max(1, tailEvents.length);
  const tailShiftKm = centroid && tailCentroid ? haversineKm(centroid, tailCentroid) : 0;
  const clusterSignal = dominantRatio >= 0.62 && spreadKm <= 160;
  const repeatSignal = previousBucket
    && previousBucket.dominantRegion === dominantRegion
    && dominantRatio >= 0.55
    && previousBucket.dominantRatio >= 0.55;
  const hiddenPatternSignal = repeatSignal && Number.isFinite(centroidShiftKm) && centroidShiftKm <= 120;
  const lateTighteningSignal = tailSpreadKm <= spreadKm * 0.72;
  const lateDominanceSignal = tailDominantRegion === dominantRegion && tailDominantRatio >= Math.max(0.6, dominantRatio);
  const handoffSignal = tailDominantRegion !== dominantRegion && tailDominantRatio >= 0.5;
  const corridorSignal = spreadKm >= 260 && uniqueRegionCount >= 3 && switchRate >= 0.45;

  let densityText = "yayvan";
  if (spreadKm <= 80) densityText = "çok sıkı";
  else if (spreadKm <= 150) densityText = "sıkı";
  else if (spreadKm <= 240) densityText = "orta yoğun";

  const trendParts = [];
  if (countDelta != null) {
    if (countDelta >= 4) trendParts.push("olay hacmi belirgin biçimde arttı");
    else if (countDelta >= 2) trendParts.push("olay sayısı hafif arttı");
    else if (countDelta <= -4) trendParts.push("olay hacmi belirgin biçimde sakinleşti");
    else if (countDelta <= -2) trendParts.push("olay sayısı hafif azaldı");
  }
  if (previousSpreadKm != null) {
    if (spreadKm <= previousSpreadKm * 0.72) trendParts.push("cluster sıkılaşıyor");
    else if (spreadKm >= previousSpreadKm * 1.28) trendParts.push("cluster çözülüp yayılıyor");
  }
  if (Number.isFinite(centroidShiftKm) && centroidShiftKm >= 180) {
    trendParts.push(`odak yaklaşık ${Math.round(centroidShiftKm)} km sıçradı`);
  } else if (Number.isFinite(centroidShiftKm) && centroidShiftKm <= 90 && previousBucket) {
    trendParts.push("odak neredeyse aynı hacimde kaldı");
  }
  if (lateTighteningSignal) {
    trendParts.push("üst katmanda hacim belirgin biçimde daralıyor");
  }
  if (handoffSignal) {
    trendParts.push(`son bölümde odak ${tailDominantRegion} tarafına kayıyor`);
  } else if (lateDominanceSignal && tailShiftKm <= 120) {
    trendParts.push("son bölümde aynı cluster ekseni korunuyor");
  }
  if (corridorSignal && tailShiftKm >= 120) {
    trendParts.push("3D akış bir geçiş koridoru gibi uzuyor");
  }

  const seed = (
    events.length * 13
    + Math.round(spreadKm)
    + uniqueRegionCount * 17
    + Math.round((dominantRatio + secondaryRatio) * 100)
  );

  const leadSentence = pickVariant(seed, [
    `${events.length} olaylık bu 12 saatlik blokta ana yük ${dominantRegion} çevresinde toplandı.`,
    `Bu 12 saatlik kesitte akışın merkezi ${dominantRegion} tarafında kaldı ve toplam ${events.length} olay üretildi.`,
    `${events.length} olayın bıraktığı hacim, ilk bakışta ${dominantRegion} eksenini öne çıkarıyor.`,
  ]);

  let patternSentence = "";
  if (hiddenPatternSignal) {
    patternSentence = pickVariant(seed + 3, [
      `${dominantRegion} iki pencere boyunca merkezde kaldı; tekrar eden bu odak erken-uyarı açısından güçlü bir öncü cluster adayı gibi duruyor.`,
      `Aynı 3D odak ikinci pencereye de taşmış durumda; ${dominantRegion} çevresindeki bu tekrar, gizli ama ısrarcı bir cluster izi veriyor.`,
      `${dominantRegion} ardışık iki blokta da çözülmedi; bu süreklilik, sıradan saçılmadan çok kalıcı bir çekirdeğe işaret ediyor.`,
    ]);
  } else if (repeatSignal && clusterSignal) {
    patternSentence = pickVariant(seed + 5, [
      `${dominantRegion} yeniden öne çıkıp sıkı kaldı; grafik tek çekirdekli bir tekrar üretmiş görünüyor.`,
      `Aynı zon bu blokta da sıkı biçimde korunuyor; tekrar eden 3D çekirdek artık rastlantısal görünmüyor.`,
      `${dominantRegion} ikinci kez dar bir hacimde öne çıktı; bu, aynı cluster'a geri dönme eğilimini güçlendiriyor.`,
    ]);
  } else if (clusterSignal) {
    patternSentence = pickVariant(seed + 7, [
      `Hacim ${densityText}; tek başına belirginleşen bir cluster en çok ${dominantRegion} içinde okunuyor.`,
      `${dominantRegion} içinde ${densityText} bir çekirdek var; bu blokta dağınıklıktan çok sıkışma baskın.`,
      `Bu pencere, ${dominantRegion} çevresinde tek odaklı bir sıkışma üretmiş durumda.`,
    ]);
  } else if (secondaryRegion && secondaryRatio >= 0.28) {
    patternSentence = pickVariant(seed + 11, [
      `${dominantRegion} baskın kalsa da ${secondaryRegion} ikinci bir kol açıyor; akış iki cluster arasında paylaşılıyor.`,
      `Tek odak yerine çift omurgalı bir desen var: ${dominantRegion} birinci, ${secondaryRegion} ise güçlü ikinci kanal.`,
      `${secondaryRegion}, ${dominantRegion} arkasında sıradan bir artçı değil; belirgin bir ikinci odak gibi davranıyor.`,
    ]);
  } else if (handoffSignal) {
    patternSentence = pickVariant(seed + 13, [
      `Blok geneli ${dominantRegion} ağırlıklı olsa da son bölümde akış ${tailDominantRegion} tarafına el değiştiriyor.`,
      `Başlangıç ve son bölüm aynı zonu göstermiyor; 3D örüntü kapanışı ${tailDominantRegion} eksenine kaydırmış.`,
      `Bu pencere tek cluster üretmekten çok bir odak devri sergiliyor; son söz ${tailDominantRegion} tarafında.`,
    ]);
  } else if (dominantRatio < 0.45 && uniqueRegionCount >= 4 && switchRate >= 0.55) {
    patternSentence = pickVariant(seed + 17, [
      "Tek bir öncü cluster ayrışmıyor; akış renkler arasında dolaşan dağınık bir zincir gibi davranıyor.",
      "Lejant tarafında net bir hakim renk yok; 3D dağılım henüz tek odaklı bir sinyal vermekten uzak.",
      "Bu blokta gizli örüntü varsa bile tek merkezli değil; çok zonlu geçişler baskın kalmış.",
    ]);
  } else if (corridorSignal) {
    patternSentence = pickVariant(seed + 19, [
      "Dağılım noktasal bir kümeye kapanmıyor; uzayan bir geçiş koridoru hissi veriyor.",
      "Noktalar tek merkezde düğümlenmek yerine bir rota çiziyor; bu blok daha çok koridor davranışı taşıyor.",
      "Bu hacim cluster'dan çok yönlü bir akışa benziyor; 3D izlekte uzayan bir omurga var.",
    ]);
  } else {
    patternSentence = pickVariant(seed + 23, [
      `${dominantRegion} önde, fakat çekirdek tam kapanmıyor; sistem hâlâ hareketli bir geçiş alanı üretiyor.`,
      `${dominantRegion} baskın olsa da hacim tam oturmuş değil; güçlü bir öncü çekirdek yerine kararsız bir alan hissi var.`,
      `Grafik ${dominantRegion} tarafına eğiliyor, ancak örüntü henüz sert bir cluster kararı vermiyor.`,
    ]);
  }

  let closingSentence = "";
  if (lateTighteningSignal && lateDominanceSignal) {
    closingSentence = ` Son bölümde ${tailDominantRegion} tarafında sıkılaşma var; eğer aynı odak bir sonraki blokta da korunursa erken-uyarı açısından daha dikkat çekici hale gelir.`;
  } else if (handoffSignal) {
    closingSentence = ` Son saatlerde ${tailDominantRegion} öne çıktığı için iz sürülmesi gereken yer artık pencere başındaki zon değil, kapanıştaki yeni odak.`;
  } else if (dominantRatio < 0.45 && uniqueRegionCount >= 4) {
    closingSentence = " Şimdilik gizli örüntü arayışı tek cluster yerine geçiş frekansında ve renk değişim hızında daha anlamlı görünüyor.";
  } else if (secondaryRegion && secondaryRatio >= 0.28) {
    closingSentence = ` ${secondaryRegion} ikinci kanal olarak kalmaya devam ederse bu ikili yapı, gelecekteki sıçramaların ana anahtarı olabilir.`;
  }

  const trendSentence = trendParts.length ? ` ${trendParts.join("; ")}.` : "";
  const summary = `${leadSentence} ${patternSentence}${trendSentence}${closingSentence}`.trim();

  return {
    count: events.length,
    centroid,
    spreadKm: Number(spreadKm.toFixed(1)),
    dominantRegion,
    dominantRatio: Number(dominantRatio.toFixed(2)),
    secondaryRegion,
    secondaryRatio: Number(secondaryRatio.toFixed(2)),
    uniqueRegionCount,
    switchRate: Number(switchRate.toFixed(2)),
    centroidShiftKm: Number.isFinite(centroidShiftKm) ? Number(centroidShiftKm.toFixed(1)) : null,
    clusterSignal,
    repeatSignal,
    hiddenPatternSignal,
    summary,
  };
}

function hourlySpaceTimeInsights(hours = 4, maxEvents = 144, bucketHours = 12) {
  const source = state.events.slice(-maxEvents);
  if (source.length < 4) return [];
  const buckets = new Map();
  for (const event of source) {
    const hourStart = toTurkeyWindowStart(event.timestamp, bucketHours);
    if (!buckets.has(hourStart)) buckets.set(hourStart, []);
    buckets.get(hourStart).push(event);
  }
  const orderedHours = [...buckets.keys()].sort((a, b) => a - b).slice(-hours);
  const raw = orderedHours.map((hourStart) => ({
    hourStart,
    hourLabel: formatTurkeyWindowLabel(hourStart, bucketHours),
    events: buckets.get(hourStart).sort((a, b) => a.timestamp - b.timestamp),
  }));
  const enriched = raw.map((bucket, index) => {
    const previous = index > 0 ? raw[index - 1]._summary : null;
    const summary = summarizeSpaceTimeBucket(bucket.events, previous);
    bucket._summary = summary;
    return {
      hourLabel: bucket.hourLabel,
      count: summary.count,
      dominantRegion: summary.dominantRegion,
      dominantRatio: summary.dominantRatio,
      uniqueRegionCount: summary.uniqueRegionCount,
      switchRate: summary.switchRate,
      spreadKm: summary.spreadKm,
      centroidShiftKm: summary.centroidShiftKm,
      summary: summary.summary,
    };
  });
  return enriched.reverse();
}

function spaceTimeSequences(windows = [1, 3, 6, 12], maxEvents = 240) {
  const source = state.events.slice(-maxEvents);
  const latest = source.at(-1);
  if (!latest) return [];
  return windows.map((hours) => {
    const cutoff = latest.timestamp - (hours * 3_600_000);
    const items = source
      .filter((event) => event.timestamp >= cutoff)
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((event, index, array) => {
        const region = classifyRegion(event);
        const previous = index > 0 ? array[index - 1] : null;
        const deltaMinutes = previous ? Number(((event.timestamp - previous.timestamp) / 60_000).toFixed(1)) : null;
        return {
          id: event.id,
          order: index + 1,
          timeLabel: formatTurkeyClock(event.timestamp),
          region,
          magnitude: Number(event.magnitude.toFixed(1)),
          depth: Number(event.depth.toFixed(1)),
          latitude: Number(event.latitude.toFixed(3)),
          longitude: Number(event.longitude.toFixed(3)),
          deltaMinutes,
        };
      });
    return {
      hours,
      label: `${hours} saat`,
      count: items.length,
      latestTimeLabel: formatTurkeyClock(latest.timestamp),
      items,
    };
  });
}

function enrichPredictionUncertainty(prediction) {
  if (!prediction) return prediction;
  const confidenceThreshold = computeAdaptiveConfidenceThreshold();
  const calibratedConfidence = calibratedConfidenceForPrediction(prediction);
  if (
    Number.isFinite(prediction.predictedRadiusKm)
    && Number.isFinite(prediction.predictedMajorAxisKm)
    && Number.isFinite(prediction.predictedMinorAxisKm)
  ) {
    return {
      ...prediction,
      confidence: Number.isFinite(calibratedConfidence) ? calibratedConfidence : prediction.confidence,
      confidenceClass: confidenceClassForScore(
        Number.isFinite(calibratedConfidence) ? calibratedConfidence : prediction.confidence,
        confidenceThreshold,
      ),
    };
  }
  const uncertainty = uncertaintyMetrics(prediction.regionConfidence, prediction.region);
  return {
    ...prediction,
    rawConfidence: Number.isFinite(prediction.rawConfidence) ? prediction.rawConfidence : uncertainty.rawConfidence,
    confidence: Number.isFinite(calibratedConfidence)
      ? calibratedConfidence
      : (Number.isFinite(prediction.confidence) ? prediction.confidence : uncertainty.confidence),
    confidenceClass: confidenceClassForScore(
      Number.isFinite(calibratedConfidence)
        ? calibratedConfidence
        : (Number.isFinite(prediction.confidence) ? prediction.confidence : uncertainty.confidence),
      confidenceThreshold,
    ),
    predictedRadiusKm: Number.isFinite(prediction.predictedRadiusKm) ? prediction.predictedRadiusKm : uncertainty.radiusKm,
    predictedMajorAxisKm: Number.isFinite(prediction.predictedMajorAxisKm) ? prediction.predictedMajorAxisKm : uncertainty.majorAxisKm,
    predictedMinorAxisKm: Number.isFinite(prediction.predictedMinorAxisKm) ? prediction.predictedMinorAxisKm : uncertainty.minorAxisKm,
    predictedAngleDeg: Number.isFinite(prediction.predictedAngleDeg) ? prediction.predictedAngleDeg : uncertainty.angleDeg,
  };
}

function activePredictionIsStale() {
  const activePrediction = state.predictions[0];
  const latestEvent = state.events.at(-1);
  if (!activePrediction || !latestEvent?.id) return false;
  return activePrediction.basedOnEventId !== latestEvent.id;
}

function canGeneratePredictionNow() {
  return Boolean(model) && state.events.length > LOOKBACK;
}

async function ensureActivePrediction({ persist = true } = {}) {
  if (!canGeneratePredictionNow() || state.predictions.length) return false;
  setActivePredictionForReferenceIndex(state.events.length - 1);
  if (persist) await savePredictionState();
  return true;
}

async function updatePredictions() {
  if (state.events.length <= LOOKBACK) {
    state.previousPrediction = state.predictions[0] || state.previousPrediction;
    state.predictions = [];
    await savePredictionState();
    return;
  }
  setActivePredictionForReferenceIndex(state.events.length - 1);
  await savePredictionState();
}

function publicState() {
  return {
      config: {
        startDate: `${START_DATE_TURKEY} TSİ`,
        mainCatalogYears: MAIN_CATALOG_YEARS,
        largeEventTrainingStartDate: `${LARGE_EVENT_START_DATE_TURKEY} TSİ`,
        largeEventCatalogYears: LARGE_EVENT_CATALOG_YEARS,
        queryStartDateUtc: turkeyDateToAfadUtc(START_DATE_TURKEY),
        fetchIntervalMs: FETCH_INTERVAL_MS,
        lookback: LOOKBACK,
        modelVersion: MODEL_VERSION,
        modelType: "GRU + region -> grid-cell -> global decode + normalize spatial loss",
        summaryWindows: SUMMARY_WINDOWS,
        eventFeatureSize: EVENT_FEATURE_SIZE,
        eventFeatureNames: EVENT_FEATURE_NAMES,
        motifFeatureSize: MOTIF_FEATURE_SIZE,
        motifFeatureNames: MOTIF_FEATURE_NAMES,
        gridRows: GRID_ROWS,
        gridCols: GRID_COLS,
        summaryFeatureSize: SUMMARY_FEATURE_SIZE,
        summaryFeatureNames: SUMMARY_FEATURE_NAMES,
        summaryInputSize: SUMMARY_INPUT_SIZE,
        largeEventMagnitudeThreshold: LARGE_EVENT_MAGNITUDE_THRESHOLD,
        largeEventLookaheadHours: LARGE_EVENT_LOOKAHEAD_HOURS,
        source: AFAD_URL,
      },
      metrics: {
        ...state.metrics,
        recentPerformance: recentPerformanceMetrics(),
        largeEventPerformance: largeEventPerformanceMetrics(),
        regionPerformance: regionPerformanceMetrics(),
        regionTransitions: regionTransitionMetrics(),
        regionPerformanceByWindow: {
          12: regionPerformanceMetrics(12),
          60: regionPerformanceMetrics(60),
          all: regionPerformanceMetrics("all"),
        },
        magnitudePerformanceByWindow: {
          12: magnitudePerformanceMetrics(12),
          60: magnitudePerformanceMetrics(60),
          all: magnitudePerformanceMetrics("all"),
        },
        depthPerformanceByWindow: {
          12: depthPerformanceMetrics(12),
          60: depthPerformanceMetrics(60),
          all: depthPerformanceMetrics("all"),
        },
        regionTransitionsByWindow: {
          12: regionTransitionMetrics(12),
          60: regionTransitionMetrics(60),
          all: regionTransitionMetrics("all"),
        },
        eventCount: state.events.length,
        largeEventTrainingEventCount: state.largeEventTrainingEvents.length,
        currentDatasetSampleCount: currentDatasetSampleCount(),
        largeEventDatasetSampleCount: currentDatasetSampleCount(state.largeEventTrainingEvents.length ? state.largeEventTrainingEvents : state.events),
        initialTrainingRequired: initialHistoricalTrainingRequired(),
      sessionElapsedMs: Date.now() - sessionStartedAt.getTime(),
      totalLearningMs: currentLearningMs(),
    },
    latestEvents: state.events.slice(-80).reverse(),
    predictionCount: state.predictions.length,
    predictions: state.predictions.map(enrichPredictionUncertainty),
    previousPrediction: enrichPredictionUncertainty(state.previousPrediction),
    lastMatch: state.lastMatch,
    recentMatches: state.recentMatches,
    comparisonHistory: state.comparisonHistory,
    spaceTimeInsights: hourlySpaceTimeInsights(),
    spaceTimeSequences: spaceTimeSequences(),
    regionCombinationPatterns: state.regionCombinationPatterns,
    showMapHistory: state.showMapHistory,
  };
}

function publicStatusLite() {
  return {
    receivedAt: new Date().toISOString(),
    metrics: {
      status: state.metrics.status,
      error: state.metrics.error,
      lastAction: state.metrics.lastAction,
      lastTrainingAt: state.metrics.lastTrainingAt,
      lastTrainingMs: state.metrics.lastTrainingMs,
      totalTrainingRunCount: state.metrics.totalTrainingRunCount,
      lastRunTrainedSamples: state.metrics.lastRunTrainedSamples,
      totalTrainedSamples: state.metrics.totalTrainedSamples,
      modelMemory: state.metrics.modelMemory,
      trainingProgress: state.metrics.trainingProgress,
      eventCount: state.events.length,
      freshEvents: state.metrics.freshEvents,
      initialTrainingRequired: initialHistoricalTrainingRequired(),
    },
  };
}

function broadcast() {
  io?.emit("state", publicState());
}

async function cycle() {
  ensureTrainingFlagConsistency();
  if (afadCycleSuspended || trainingInProgress || trainingStartPending || localCatalogTrainingActive) return;
  if (initialHistoricalTrainingRequired()) {
    state.metrics.status = "ready_for_training";
    state.metrics.error = null;
    state.metrics.lastAction = "Canlı AFAD takibi beklemede. Önce geçmiş katalogla ilk eğitim tamamlanacak.";
    broadcast();
    return;
  }
  try {
    const freshCount = await fetchAfadEvents();
    await trainOnlyWhenFresh(freshCount);
    const largeEventSignalsChanged = reconcileLargeEventSignals();
    if (largeEventSignalsChanged) {
      await savePredictionState();
    }
    if (!state.predictions.length && canGeneratePredictionNow()) {
      const created = await ensureActivePrediction();
      if (created) {
        state.metrics.status = "learning";
        state.metrics.error = null;
        state.metrics.lastAction = freshCount
          ? "Yeni veri sonrası aktif tahmin yeniden oluşturuldu."
          : "Yeni AFAD olayı yok; mevcut model belleğiyle aktif tahmin oluşturuldu.";
      }
    }
    broadcast();
  } catch (error) {
    state.metrics.status = "error";
    state.metrics.error = error.message;
    state.metrics.lastAction = `AFAD kontrol başarısız: ${error.message}`;
    broadcast();
    console.error(error);
  }
}

await loadRuntimeState();
await loadEvents();
await loadLargeEventTrainingEvents();
await loadModel();
await loadPredictionState();

const app = express();
const server = createServer(app);
io = new Server(server, { cors: { origin: "*" } });

app.get("/api/state", async (req, res) => {
  ensureTrainingFlagConsistency();
  if (!trainingInProgress && !trainingStartPending && !state.predictions.length && canGeneratePredictionNow()) {
    try {
      const created = await ensureActivePrediction();
      if (created) {
        state.metrics.status = initialHistoricalTrainingRequired() ? "ready_for_training" : "learning";
        state.metrics.error = null;
        if (!state.metrics.lastAction || state.metrics.lastAction.includes("eğitim yapıldı")) {
          state.metrics.lastAction = "Mevcut model belleğiyle aktif tahmin oluşturuldu.";
        }
        await savePredictionState();
        await saveRuntimeState();
      }
    } catch (error) {
      state.metrics.error = error.message;
    }
  }
  res.json(publicState());
});
app.get("/api/status-lite", (req, res) => {
  ensureTrainingFlagConsistency();
  res.json(publicStatusLite());
});
app.post("/api/train-existing", async (req, res) => {
  ensureTrainingFlagConsistency();
  if (trainingInProgress || trainingStartPending || localCatalogTrainingActive) {
    res.json({
      metrics: {
        status: state.metrics.status,
        lastAction: state.metrics.lastAction,
        error: state.metrics.error,
      },
      trainResult: { started: false, trained: false, freshCount: 0, reason: "already_training" },
    });
    return;
  }
  state.metrics.error = null;
  state.metrics.status = "training";
  state.metrics.lastAction = "Mevcut katalog verileriyle arka planda eğitim başlatıldı.";
  trainingStartPending = true;
  localCatalogTrainingActive = true;
  afadCycleSuspended = true;
  broadcast();
  res.json({
    metrics: {
      status: state.metrics.status,
      lastAction: state.metrics.lastAction,
      error: state.metrics.error,
    },
    trainResult: { started: true, background: true, trained: false, freshCount: 0, reason: "background_started" },
  });
  setTimeout(() => {
    void trainOnlyWhenFresh(0, { manual: true, forceInitial: true, localOnly: true })
      .then(async (result) => {
        trainingStartPending = false;
        localCatalogTrainingActive = false;
        afadCycleSuspended = initialHistoricalTrainingRequired();
        if (!trainingInProgress && state.metrics.status === "training") {
          state.metrics.status = initialHistoricalTrainingRequired() ? "ready_for_training" : "learning";
        }
        state.metrics.lastAction = result.trained
          ? "Mevcut katalog verileriyle eğitim yapıldı."
          : (state.metrics.lastAction || "Mevcut verilerle eğitim yapılmadı.");
        await saveRuntimeState();
        await savePredictionState();
        broadcast();
      })
      .catch(async (error) => {
        trainingStartPending = false;
        localCatalogTrainingActive = false;
        afadCycleSuspended = initialHistoricalTrainingRequired();
        state.metrics.status = "learning";
        state.metrics.error = error.message;
        state.metrics.lastAction = `Yerel eğitim başarısız: ${error.message}`;
        await saveRuntimeState();
        broadcast();
      });
  }, 0);
});
app.post("/api/train", async (req, res) => {
  ensureTrainingFlagConsistency();
  if (initialHistoricalTrainingRequired()) {
    state.metrics.status = "ready_for_training";
    state.metrics.error = null;
    state.metrics.lastAction = "Önce 'Mevcut verilerle öğren' ile geçmiş katalog eğitimi tamamlanmalı.";
    broadcast();
    res.json({ ...publicState(), trainResult: { trained: false, freshCount: 0, reason: "initial_training_required" } });
    return;
  }
  if (trainingInProgress || trainingStartPending) {
    state.metrics.status = trainingInProgress ? "training" : "fetching";
    state.metrics.error = null;
    state.metrics.lastAction = trainingInProgress
      ? "Yeni AFAD verileriyle eğitim zaten sürüyor."
      : "AFAD canlı kataloğu arka planda zaten kontrol ediliyor.";
    broadcast();
    res.json({ ...publicState(), trainResult: { trained: false, freshCount: 0, reason: "already_training" } });
    return;
  }

  trainingStartPending = true;
  state.metrics.status = "fetching";
  state.metrics.error = null;
  state.metrics.lastAction = "AFAD canlı kataloğu arka planda kontrol ediliyor.";
  broadcast();
  res.json({ ...publicState(), trainResult: { trained: false, freshCount: 0, reason: "background_started" } });

  setTimeout(() => {
    (async () => {
      try {
        ensureTrainingFlagConsistency();
        if (resetPending && state.events.length > LOOKBACK + 2) {
          const result = await trainOnlyWhenFresh(0, { manual: true, forceInitial: true });
          state.metrics.lastAction = result?.trained
            ? "Sıfırlama sonrası eğitim arka planda tamamlandı."
            : (state.metrics.lastAction || "Sıfırlama sonrası eğitim atlandı.");
          await saveRuntimeState();
          broadcast();
          return;
        }
        const freshCount = await fetchAfadEvents();
        const result = await trainOnlyWhenFresh(freshCount, { manual: false, forceInitial: false });
        state.metrics.lastAction = freshCount > 0
          ? `${freshCount} yeni AFAD olayı bulundu${result?.trained ? " ve hafif eğitim yapıldı." : "."}`
          : "Yeni AFAD olayı yok; eğitim atlandı.";
        await saveRuntimeState();
        broadcast();
      } catch (error) {
        state.metrics.status = "learning";
        state.metrics.error = error.message;
        state.metrics.lastAction = `AFAD kontrol başarısız: ${error.message}`;
        await saveRuntimeState();
        broadcast();
      } finally {
        trainingStartPending = false;
      }
    })();
  }, 0);
});
app.post("/api/map/clear", async (req, res) => {
  state.previousPrediction = null;
  state.lastMatch = null;
  state.recentMatches = [];
  state.showMapHistory = false;
  state.metrics.lastMatchDistanceKm = null;
  await savePredictionState();
  broadcast();
  res.json(publicState());
});
app.post("/api/region-combinations/refresh", async (req, res) => {
  await refreshRegionCombinationPatterns();
  broadcast();
  res.json(publicState());
});
app.post("/api/evaluation/reset", async (req, res) => {
  const result = await resetEvaluationHistory();
  broadcast();
  res.json({ ...publicState(), resetResult: result });
});
app.post("/api/confusion/reset", async (req, res) => {
  const result = await resetConfusionHistory();
  broadcast();
  res.json({ ...publicState(), resetResult: result });
});
app.post("/api/reset", async (req, res) => {
  const result = await resetModelAndIndicators();
  broadcast();
  res.json({ ...publicState(), resetResult: result });
});
app.get("/api/health", (req, res) => res.json({ ok: true, status: state.metrics.status }));

io.on("connection", (socket) => {
  socket.emit("state", publicState());
});

server.listen(PORT, () => {
  console.log(`API: http://localhost:${PORT}`);
  console.log(`UI:  http://localhost:${VITE_PORT}`);
});

async function bootAfterListen() {
  if (mainCatalogNeedsRefresh()) {
    state.metrics.status = "fetching";
    state.metrics.lastAction = "Ana AFAD kataloğu 2 yıla ayarlanıyor.";
    broadcast();
    await ensureMainCatalogEvents();
  }
  reconcileLargeEventSignals();
  if (activePredictionIsStale() && state.events.length > LOOKBACK + 2) {
    await closeActivePredictionIfPossible();
    await updatePredictions();
  }
  await ensureActivePrediction();
  if (!state.regionCombinationPatterns.length && state.events.length >= 4) {
    await refreshRegionCombinationPatterns();
  }
  evaluateModel();
  if (!trainingInProgress && !trainingStartPending) {
    resetTrainingProgress();
  }
  if (state.metrics.modelMemory === "new" && state.events.length > LOOKBACK + 2) {
    state.predictions = [];
    state.metrics.status = "ready_for_training";
    state.metrics.lastAction = `${MAIN_CATALOG_YEARS} yıllık katalog yüklendi. Önce geçmiş katalogla ilk eğitim yapılacak; canlı AFAD takibi sonra başlayacak.`;
  } else if (!state.predictions.length && canGeneratePredictionNow()) {
    await ensureActivePrediction();
    if (state.predictions.length && !state.metrics.lastAction) {
      state.metrics.lastAction = "Kayıtlı model belleğiyle aktif tahmin hazırlandı.";
    }
  }
  await savePredictionState();
  await saveRuntimeState();
  broadcast();
  if (!initialHistoricalTrainingRequired()) {
    afadCycleSuspended = false;
    cycle();
  } else {
    afadCycleSuspended = true;
  }
}

bootAfterListen().catch((error) => {
  state.metrics.status = "error";
  state.metrics.error = error.message;
  console.error(error);
  broadcast();
});

setInterval(() => {
  if (afadCycleSuspended || localCatalogTrainingActive || trainingInProgress || trainingStartPending || initialHistoricalTrainingRequired()) {
    return;
  }
  cycle();
}, FETCH_INTERVAL_MS);
setInterval(() => {
  saveRuntimeState().catch((error) => console.error(`Runtime state could not be saved: ${error.message}`));
}, 10_000);

process.on("unhandledRejection", (reason) => {
  const message = reason instanceof Error ? reason.message : String(reason);
  state.metrics.status = trainingInProgress ? "training" : (initialHistoricalTrainingRequired() ? "ready_for_training" : "learning");
  state.metrics.error = message;
  state.metrics.lastAction = `Beklenmeyen arka plan hatası yakalandı: ${message}`;
  console.error("Unhandled rejection:", reason);
  saveRuntimeState().catch(() => {});
  broadcast();
});

process.on("uncaughtException", (error) => {
  state.metrics.status = trainingInProgress ? "training" : (initialHistoricalTrainingRequired() ? "ready_for_training" : "learning");
  state.metrics.error = error.message;
  state.metrics.lastAction = `Beklenmeyen backend hatası yakalandı: ${error.message}`;
  console.error("Uncaught exception:", error);
  saveRuntimeStateSync();
  broadcast();
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    saveRuntimeStateSync();
    process.exit(0);
  });
}

