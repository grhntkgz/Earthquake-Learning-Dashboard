import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BrainCircuit,
  DatabaseZap,
  MapPin,
  Play,
  Radar,
  RotateCcw,
  SatelliteDish,
  TriangleAlert,
} from "lucide-react";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Bar, Line, Scatter } from "react-chartjs-2";
import * as THREE from "three";
import "./styles.css";
import {
  REGION_COLORS,
  REGION_LABELS,
  REGION_ZONES,
  classifyRegion,
} from "../shared/regions.js";

function regionColorCss(region) {
  const color = REGION_COLORS[region];
  if (!Number.isFinite(color)) return "#808080";
  return `#${color.toString(16).padStart(6, "0")}`;
}

function displayRegionForPrediction(prediction) {
  if (!prediction) return "-";
  const derivedRegion = classifyRegion(prediction);
  if (derivedRegion && derivedRegion !== "Diğer") return derivedRegion;
  return prediction.region || "-";
}

const pulsePlugin = {
  id: "latestPulse",
  afterDraw(chart) {
    if (!chart.options.plugins?.latestPulse?.enabled) return;
    const datasetIndex = (label, occurrence = 0) => {
      let seen = -1;
      return chart.data.datasets.findIndex((dataset) => {
        if (dataset.label !== label) return false;
        seen += 1;
        return seen === occurrence;
      });
    };
    const latestPoint = chart.getDatasetMeta(datasetIndex("En son gerçek olay"))?.data?.[0];
    const matchPairs = [
      [
        chart.getDatasetMeta(datasetIndex("Eski tahmin", 0))?.data?.[0],
        chart.getDatasetMeta(datasetIndex("Eski gerçekleşen olay", 0))?.data?.[0],
        "rgba(100, 116, 139, 0.52)",
      ],
      [
        chart.getDatasetMeta(datasetIndex("Eski tahmin", 1))?.data?.[0],
        chart.getDatasetMeta(datasetIndex("Eski gerçekleşen olay", 1))?.data?.[0],
        "rgba(100, 116, 139, 0.32)",
      ],
    ];
    if (!latestPoint) return;
    chart.options.plugins.latestPulse.onPosition?.({ x: latestPoint.x, y: latestPoint.y });
    const ctx = chart.ctx;
    ctx.save();
    for (const [predicted, actual, color] of matchPairs) {
      if (!predicted || !actual) continue;
      ctx.beginPath();
      ctx.setLineDash([5, 5]);
      ctx.moveTo(predicted.x, predicted.y);
      ctx.lineTo(actual.x, actual.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  },
};

const regionLabelPlugin = {
  id: "regionLabels",
  afterDraw(chart) {
    if (!chart.options.plugins?.regionLabels?.enabled) return;
    const labels = chart.options.plugins.regionLabels.labels || [];
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const ctx = chart.ctx;
    const chartWidth = Math.max(0, chart.chartArea?.right - chart.chartArea?.left || 0);
    const compact = chartWidth > 0 && chartWidth < 620;
    const fontSize = compact ? 10 : 12;
    ctx.save();
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const label of labels) {
      const x = xScale.getPixelForValue(label.x);
      const y = yScale.getPixelForValue(label.y);
      const text = compact && label.shortText ? label.shortText : label.text;
      ctx.fillStyle = "rgba(71, 85, 105, 0.72)";
      ctx.fillText(text, x, y);
    }
    ctx.restore();
  },
};

const regionFillPlugin = {
  id: "regionFill",
  beforeDatasetsDraw(chart) {
    if (!chart.options.plugins?.regionFill?.enabled) return;
    const regions = chart.options.plugins.regionFill.regions || [];
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const ctx = chart.ctx;
    ctx.save();
    for (const region of regions) {
      const segments = region.segments || [region.points];
      for (const segment of segments) {
        if (!segment?.length) continue;
        ctx.beginPath();
        segment.forEach((point, index) => {
          const x = xScale.getPixelForValue(point.x);
          const y = yScale.getPixelForValue(point.y);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = region.backgroundColor;
        ctx.fill();
      }
    }
    ctx.restore();
  },
};

const predictionUncertaintyPlugin = {
  id: "predictionUncertainty",
  beforeDatasetsDraw(chart) {
    const config = chart.options.plugins?.predictionUncertainty;
    if (!config?.enabled || !config.prediction || !Number.isFinite(config.prediction.predictedRadiusKm)) return;
    const {
      latitude,
      longitude,
      predictedRadiusKm,
      predictedMajorAxisKm,
      predictedMinorAxisKm,
      predictedAngleDeg,
      confidenceClass,
    } = config.prediction;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    const x = xScale.getPixelForValue(longitude);
    const y = yScale.getPixelForValue(latitude);
    const majorKm = predictedMajorAxisKm || predictedRadiusKm;
    const minorKm = predictedMinorAxisKm || predictedRadiusKm;
    const angleRad = ((predictedAngleDeg || 0) * Math.PI) / 180;
    const kmToLon = (kmEastWest) => kmEastWest / (111 * Math.max(0.2, Math.cos((latitude * Math.PI) / 180)));
    const kmToLat = (kmNorthSouth) => kmNorthSouth / 111;
    const pixelVectorForAxis = (axisKm, axisAngleRad) => {
      const dxKm = axisKm * Math.cos(axisAngleRad);
      const dyKm = axisKm * Math.sin(axisAngleRad);
      const lonDelta = kmToLon(dxKm);
      const latDelta = kmToLat(dyKm);
      return {
        x: xScale.getPixelForValue(longitude + lonDelta) - x,
        y: yScale.getPixelForValue(latitude + latDelta) - y,
      };
    };
    const majorVector = pixelVectorForAxis(majorKm, angleRad);
    const minorVector = pixelVectorForAxis(minorKm, angleRad + Math.PI / 2);
    const majorLength = Math.hypot(majorVector.x, majorVector.y);
    const minorLength = Math.hypot(minorVector.x, minorVector.y);
    if (majorLength < 1 || minorLength < 1) return;
    const pixelRotation = Math.atan2(majorVector.y, majorVector.x);
    const styles = {
      "Yüksek": {
        fill: "rgba(20, 184, 166, 0.16)",
        stroke: "rgba(13, 148, 136, 0.52)",
      },
      "Orta": {
        fill: "rgba(14, 165, 233, 0.14)",
        stroke: "rgba(2, 132, 199, 0.48)",
      },
      "Düşük": {
        fill: "rgba(245, 158, 11, 0.14)",
        stroke: "rgba(217, 119, 6, 0.46)",
      },
    };
    const style = styles[confidenceClass] || styles.Orta;
    const ctx = chart.ctx;
    const { left, top, right, bottom } = chart.chartArea;
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    ctx.beginPath();
    ctx.ellipse(
      x,
      y,
      Math.max(majorLength, 8),
      Math.max(minorLength, 8),
      pixelRotation,
      0,
      Math.PI * 2,
    );
    ctx.fillStyle = style.fill;
    ctx.fill();
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  },
};

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, pulsePlugin, regionLabelPlugin, regionFillPlugin, predictionUncertaintyPlugin);

const API = "";
const TURKEY_OUTLINES = [
  [
    { x: 36.913127, y: 41.335358 },
    { x: 38.347665, y: 40.948586 },
    { x: 39.512607, y: 41.102763 },
    { x: 40.373433, y: 41.013673 },
    { x: 41.554084, y: 41.535656 },
    { x: 42.619549, y: 41.583173 },
    { x: 43.582746, y: 41.092143 },
    { x: 43.752658, y: 40.740201 },
    { x: 43.656436, y: 40.253564 },
    { x: 44.400009, y: 40.005 },
    { x: 44.79399, y: 39.713003 },
    { x: 44.109225, y: 39.428136 },
    { x: 44.421403, y: 38.281281 },
    { x: 44.225756, y: 37.971584 },
    { x: 44.772699, y: 37.170445 },
    { x: 44.293452, y: 37.001514 },
    { x: 43.942259, y: 37.256228 },
    { x: 42.779126, y: 37.385264 },
    { x: 42.349591, y: 37.229873 },
    { x: 41.212089, y: 37.074352 },
    { x: 40.673259, y: 37.091276 },
    { x: 39.52258, y: 36.716054 },
    { x: 38.699891, y: 36.712927 },
    { x: 38.167727, y: 36.90121 },
    { x: 37.066761, y: 36.623036 },
    { x: 36.739494, y: 36.81752 },
    { x: 36.685389, y: 36.259699 },
    { x: 36.41755, y: 36.040617 },
    { x: 36.149763, y: 35.821535 },
    { x: 35.782085, y: 36.274995 },
    { x: 36.160822, y: 36.650606 },
    { x: 35.550936, y: 36.565443 },
    { x: 34.714553, y: 36.795532 },
    { x: 34.026895, y: 36.21996 },
    { x: 32.509158, y: 36.107564 },
    { x: 31.699595, y: 36.644275 },
    { x: 30.621625, y: 36.677865 },
    { x: 30.391096, y: 36.262981 },
    { x: 29.699976, y: 36.144357 },
    { x: 28.732903, y: 36.676831 },
    { x: 27.641187, y: 36.658822 },
    { x: 27.048768, y: 37.653361 },
    { x: 26.318218, y: 38.208133 },
    { x: 26.8047, y: 38.98576 },
    { x: 26.170785, y: 39.463612 },
    { x: 27.28002, y: 40.420014 },
    { x: 28.819978, y: 40.460011 },
    { x: 29.240004, y: 41.219991 },
    { x: 31.145934, y: 41.087622 },
    { x: 32.347979, y: 41.736264 },
    { x: 33.513283, y: 42.01896 },
    { x: 35.167704, y: 42.040225 },
    { x: 36.913127, y: 41.335358 },
  ],
  [
    { x: 27.192377, y: 40.690566 },
    { x: 26.358009, y: 40.151994 },
    { x: 26.043351, y: 40.617754 },
    { x: 26.056942, y: 40.824123 },
    { x: 26.294602, y: 40.936261 },
    { x: 26.604196, y: 41.562115 },
    { x: 26.117042, y: 41.826905 },
    { x: 27.135739, y: 42.141485 },
    { x: 27.99672, y: 42.007359 },
    { x: 28.115525, y: 41.622886 },
    { x: 28.988443, y: 41.299934 },
    { x: 28.806438, y: 41.054962 },
    { x: 27.619017, y: 40.999823 },
    { x: 27.192377, y: 40.690566 },
  ],
];
function haversineKmView(a, b) {
  const radius = 6371;
  const toRad = (degree) => (degree * Math.PI) / 180;
  const dLat = toRad(b.latitude - a.latitude);
  const dLon = toRad(b.longitude - a.longitude);
  const lat1 = toRad(a.latitude);
  const lat2 = toRad(b.latitude);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(h));
}

function formatTime(value) {
  if (!value) return "-";
  const normalized = typeof value === "string" && !/[zZ]|[+-]\d\d:\d\d$/.test(value) ? `${value}Z` : value;
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: "Europe/Istanbul",
  }).format(new Date(normalized));
}

function formatShortTime(value) {
  if (!value) return "-";
  const normalized = typeof value === "string" && !/[zZ]|[+-]\d\d:\d\d$/.test(value) ? `${value}Z` : value;
  return new Intl.DateTimeFormat("tr-TR", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Europe/Istanbul",
  }).format(new Date(normalized));
}

function formatDuration(ms) {
  if (ms == null) return "-";
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours) return `${hours} sa ${minutes} dk`;
  if (minutes) return `${minutes} dk ${seconds} sn`;
  return `${seconds} sn`;
}

function formatMinutes(minutes) {
  if (minutes == null || Number.isNaN(minutes)) return "-";
  return formatDuration(minutes * 60 * 1000);
}

function formatLoss(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return Number(value).toFixed(3);
}

function useLiveState() {
  const [liveState, setLiveState] = useState({ data: null, error: null });
  useEffect(() => {
    let socket;
    let pollTimer;
    const receiveState = (nextState) => setLiveState({ data: { ...nextState, receivedAt: Date.now() }, error: null });
    const fetchState = () =>
      fetch(`${API}/api/state`, { cache: "no-store" })
        .then((res) => (res.ok ? res.json() : null))
        .then((payload) => payload && receiveState(payload))
        .catch(() => {
          setLiveState((prev) => ({ ...prev, error: "Canlı backend durumuna ulaşılamıyor." }));
        });

    fetchState();
    pollTimer = setInterval(fetchState, 15000);
    import("socket.io-client").then(({ io }) => {
      socket = io();
      socket.on("state", receiveState);
    });
    const onVisible = () => {
      if (document.visibilityState === "visible") fetchState();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      socket?.disconnect();
      clearInterval(pollTimer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);
  return liveState;
}

function useTrainingStatus() {
  const [statusState, setStatusState] = useState({ data: null, error: null });
  useEffect(() => {
    let timer;
    const fetchStatus = () =>
      fetch(`${API}/api/status-lite`, { cache: "no-store" })
        .then((res) => (res.ok ? res.json() : null))
        .then((payload) => payload && setStatusState({ data: payload, error: null }))
        .catch(() => {
          setStatusState((prev) => ({ ...prev, error: "Kısa durum servisine ulaşılamıyor." }));
        });
    fetchStatus();
    timer = setInterval(fetchStatus, 5000);
    return () => clearInterval(timer);
  }, []);
  return statusState;
}

function Stat({ icon: Icon, label, value, detail }) {
  return (
    <div className="stat">
      <div className="stat-icon"><Icon size={18} /></div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function PerformanceGauge({ score, detail }) {
  const normalized = Math.max(0, Math.min(100, Number(score ?? 0)));
  return (
    <div className="stat gauge-stat">
      <div>
        <span>Genel performans</span>
        <strong>{score != null ? `${normalized.toFixed(0)}/100` : "-"}</strong>
        <div className="score-bar" style={{ "--score": `${normalized}%` }} aria-hidden="true" />
      </div>
    </div>
  );
}

function MiniMap({ events, predictions, recentMatches, showMapHistory }) {
  const [pulsePosition, setPulsePosition] = useState(null);
  const points = useMemo(() => {
    const previousEvent = [];
    const latestEvent = events[0]
      ? [{
          x: events[0].longitude,
          y: events[0].latitude,
          r: Math.max(8, events[0].magnitude * 5),
          label: events[0].location,
          mag: events[0].magnitude,
          kind: "En son gerçek olay",
        }]
      : [];
    const predicted = predictions.map((item) => ({
      x: item.longitude,
      y: item.latitude,
      r: 12,
      label: "Tahmin",
      mag: item.magnitude,
      kind: "Yeni tahmin",
    }));
    const latestEventId = events[0]?.id;
    const matchDatasets = (showMapHistory ? recentMatches || [] : []).slice(0, 2).map((match, index) => {
      const opacity = index === 0 ? 0.34 : 0.18;
      const actualIsLatest = Boolean(latestEventId && match.actual?.id === latestEventId);
      return {
        prediction: [{
          x: match.predicted.longitude,
          y: match.predicted.latitude,
          label: `Eski tahmin ${index + 1}`,
          mag: match.predicted.magnitude,
          kind: `Eski tahmin, hata ${match.distanceKm} km`,
        }],
        actual: [{
          x: match.actual.longitude,
          y: match.actual.latitude,
          label: match.actual.location,
          mag: match.actual.magnitude,
          kind: `Gerçekleşen olay, hata ${match.distanceKm} km`,
          hidden: actualIsLatest,
        }],
        opacity,
      };
    });
    while (matchDatasets.length < 2) {
      matchDatasets.push({ prediction: [], actual: [], opacity: 0.14 });
    }
    return { previousEvent, latestEvent, predicted, matchDatasets };
  }, [events, predictions, recentMatches, showMapHistory]);

  return (
    <div className="map-chart-wrap">
      {pulsePosition ? (
        <span
          className="latest-pulse-overlay"
          style={{ left: pulsePosition.x, top: pulsePosition.y }}
          aria-hidden="true"
        />
      ) : null}
      <Scatter
      data={{
        datasets: [
          ...TURKEY_OUTLINES.map((outline) => ({
            label: "Türkiye sınırı",
            data: outline,
            showLine: true,
            borderColor: "rgba(148, 163, 184, 0.5)",
            borderWidth: 1.25,
            pointRadius: 0,
            pointHoverRadius: 0,
            fill: false,
          })),
          {
            label: "Önceki gerçek olay",
            data: points.previousEvent,
            backgroundColor: "rgba(100, 116, 139, 0.18)",
            borderColor: "rgba(71, 85, 105, 0.34)",
            pointRadius: (ctx) => ctx.raw?.r || 6,
            pointHoverRadius: 10,
            order: 10,
          },
          {
            label: "En son gerçek olay",
            data: points.latestEvent,
            backgroundColor: "#dc2626",
            borderColor: "#7f1d1d",
            pointRadius: (ctx) => ctx.raw?.r || 10,
            pointHoverRadius: 16,
            order: 11,
          },
          {
            label: "Yeni tahmin",
            data: points.predicted,
            backgroundColor: "rgba(20, 184, 166, 0.9)",
            borderColor: "#0f766e",
            pointStyle: "triangle",
            pointRadius: 13,
            pointHoverRadius: 18,
            order: 12,
          },
          {
            label: "Eski tahmin",
            data: points.matchDatasets[0].prediction,
            backgroundColor: `rgba(20, 184, 166, ${points.matchDatasets[0].opacity})`,
            borderColor: "rgba(15, 118, 110, 0.42)",
            pointStyle: "triangle",
            pointRadius: 10,
            order: 9,
          },
          {
            label: "Eski gerçekleşen olay",
            data: points.matchDatasets[0].actual,
            backgroundColor: `rgba(100, 116, 139, ${points.matchDatasets[0].opacity + 0.08})`,
            borderColor: "rgba(71, 85, 105, 0.44)",
            pointRadius: (ctx) => (ctx.raw?.hidden ? 0 : 8),
            pointHoverRadius: (ctx) => (ctx.raw?.hidden ? 0 : 10),
            order: 8,
          },
          {
            label: "Eski tahmin",
            data: points.matchDatasets[1].prediction,
            backgroundColor: `rgba(20, 184, 166, ${points.matchDatasets[1].opacity})`,
            borderColor: "rgba(15, 118, 110, 0.24)",
            pointStyle: "triangle",
            pointRadius: 9,
            order: 7,
          },
          {
            label: "Eski gerçekleşen olay",
            data: points.matchDatasets[1].actual,
            backgroundColor: `rgba(100, 116, 139, ${points.matchDatasets[1].opacity + 0.06})`,
            borderColor: "rgba(71, 85, 105, 0.28)",
            pointRadius: (ctx) => (ctx.raw?.hidden ? 0 : 7),
            pointHoverRadius: (ctx) => (ctx.raw?.hidden ? 0 : 9),
            order: 6,
          },
        ],
      }}
      options={{
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { min: 24, max: 46, title: { display: true, text: "Boylam" } },
          y: { min: 34, max: 44, title: { display: true, text: "Enlem" } },
        },
        plugins: {
          latestPulse: {
            enabled: true,
            onPosition: (position) => {
              setPulsePosition((current) => (
                current && Math.abs(current.x - position.x) < 0.5 && Math.abs(current.y - position.y) < 0.5
                  ? current
                  : position
              ));
            },
          },
          regionFill: {
            enabled: true,
            regions: REGION_ZONES,
          },
          predictionUncertainty: {
            enabled: true,
            prediction: points.predicted[0]
              ? {
                  latitude: points.predicted[0].y,
                  longitude: points.predicted[0].x,
                  predictedRadiusKm: predictions?.[0]?.predictedRadiusKm,
                  predictedMajorAxisKm: predictions?.[0]?.predictedMajorAxisKm,
                  predictedMinorAxisKm: predictions?.[0]?.predictedMinorAxisKm,
                  predictedAngleDeg: predictions?.[0]?.predictedAngleDeg,
                  confidenceClass: predictions?.[0]?.confidenceClass,
                }
              : null,
          },
          regionLabels: {
            enabled: true,
            labels: REGION_LABELS,
          },
          legend: {
            labels: {
              color: "#263238",
              filter: (item, data) => {
                const dataset = data.datasets[item.datasetIndex];
                if (!dataset.data.length) return false;
                if (item.text === "Türkiye sınırı") return false;
                if (item.text.endsWith("bölgesi") || item.text === "Diğer bölge") return false;
                if (item.text === "Eski gerçekleşen olay") return false;
                if (item.text === "Eski tahmin") return false;
                return true;
              },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.raw.kind} - ${ctx.raw.label}: ${ctx.raw.y.toFixed(3)}, ${ctx.raw.x.toFixed(3)} ML ${ctx.raw.mag?.toFixed?.(1) ?? "-"}`,
            },
          },
        },
      }}
      />
    </div>
  );
}

function MagnitudeChart({ events }) {
  const series = events.slice(0, 40).reverse();
  return (
    <Line
      data={{
        labels: series.map((event) => formatShortTime(event.date)),
        datasets: [
          {
            label: "Büyüklük",
            data: series.map((event) => event.magnitude),
            borderColor: "#2563eb",
            backgroundColor: "#2563eb",
            tension: 0.35,
          },
          {
            label: "Derinlik / 10",
            data: series.map((event) => event.depth / 10),
            borderColor: "#f59e0b",
            backgroundColor: "#f59e0b",
            tension: 0.35,
          },
        ],
      }}
      options={{
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { position: "bottom" } },
      }}
    />
  );
}

function SpaceTimeTrail({ events }) {
  const series = events.slice(0, 30).reverse();
  if (!series.length) {
    return <p className="empty">Uzay-zaman izi için olay birikiyor.</p>;
  }

  const datasets = series.map((event, index) => {
    const age = index / Math.max(series.length - 1, 1);
    const alpha = 0.18 + (1 - age) * 0.72;
    const radius = 3 + (1 - age) * 6 + Math.max(0, event.magnitude - 1) * 1.5;
    return {
      label: event.id,
      data: [{ x: event.longitude, y: event.latitude }],
      backgroundColor: `rgba(37, 99, 235, ${alpha.toFixed(3)})`,
      borderColor: `rgba(30, 64, 175, ${(alpha + 0.08).toFixed(3)})`,
      pointRadius: radius,
      pointHoverRadius: radius + 2,
      pointStyle: "star",
    };
  });

  const pathDataset = {
    label: "İz",
    data: series.map((event) => ({ x: event.longitude, y: event.latitude })),
    showLine: true,
    borderColor: "rgba(59, 130, 246, 0.22)",
    borderWidth: 1.5,
    pointRadius: 0,
    pointHoverRadius: 0,
  };

  return (
    <Scatter
      data={{ datasets: [pathDataset, ...datasets] }}
      options={{
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { min: 24, max: 46, title: { display: true, text: "Boylam" } },
          y: { min: 34, max: 44, title: { display: true, text: "Enlem" } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw || {};
                const event = series.find((item) => item.longitude === raw.x && item.latitude === raw.y);
                if (!event) return `${raw.y}, ${raw.x}`;
                return `${formatShortTime(event.date)} - ${event.location} ML ${event.magnitude.toFixed(1)}`;
              },
            },
          },
        },
      }}
    />
  );
}

function SpaceTimeScene3D({ events }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const width = mount.clientWidth || 320;
    const height = mount.clientHeight || 280;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fbff);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 16, 34);
    camera.lookAt(0, 6, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(width, height);
    mount.innerHTML = "";
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.95));
    const light = new THREE.DirectionalLight(0xdbeafe, 1.15);
    light.position.set(12, 18, 10);
    scene.add(light);

    const lonCenter = 35;
    const latCenter = 39;
    const lonScale = 1.05;
    const latScale = 1.2;
    const timeScale = 14;
    const series = events.slice(0, 30).reverse();
    const minTimestamp = series[0]?.timestamp ?? 0;
    const maxTimestamp = series.at(-1)?.timestamp ?? minTimestamp + 1;
    const span = Math.max(1, maxTimestamp - minTimestamp);

    const grid = new THREE.GridHelper(28, 10, 0xcbd5e1, 0xe2e8f0);
    scene.add(grid);

    const frameEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(24, timeScale, 13));
    const frame = new THREE.LineSegments(
      frameEdges,
      new THREE.LineBasicMaterial({ color: 0xcbd5e1, transparent: true, opacity: 0.72 }),
    );
    frame.position.y = timeScale / 2;
    scene.add(frame);

    const points = [];
    series.forEach((event, index) => {
      const age = index / Math.max(series.length - 1, 1);
      const x = (event.longitude - lonCenter) * lonScale;
      const z = (event.latitude - latCenter) * latScale;
      const y = ((event.timestamp - minTimestamp) / span) * timeScale;
      const region = classifyRegion(event);
      points.push({ vector: new THREE.Vector3(x, y, z), event, region });

      const radius = 0.16 + (1 - age) * 0.28 + Math.max(0, event.magnitude - 1) * 0.05;
      const color = REGION_COLORS[region] || REGION_COLORS.Diğer;
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 20, 20),
        new THREE.MeshStandardMaterial({
          color,
          transparent: true,
          opacity: 0.24 + (1 - age) * 0.74,
          emissive: new THREE.Color(color),
          emissiveIntensity: (1 - age) * 0.45,
        }),
      );
      dot.position.set(x, y, z);
      scene.add(dot);
    });

    if (points.length > 1) {
      for (let i = 1; i < points.length; i += 1) {
        const previous = points[i - 1];
        const current = points[i];
        const distanceKm = haversineKmView(previous.event, current.event);
        if (distanceKm > 450) continue;
        scene.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([previous.vector, current.vector]),
          new THREE.LineBasicMaterial({
            color: REGION_COLORS[current.region] || REGION_COLORS.Diğer,
            transparent: true,
            opacity: 0.38,
          }),
        ));
      }
    }

    const latestPoint = points.at(-1)?.vector;
    if (latestPoint) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.58, 0.04, 12, 48),
        new THREE.MeshBasicMaterial({ color: 0xdc2626, transparent: true, opacity: 0.82 }),
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.copy(latestPoint);
      scene.add(ring);
    }

    const clock = new THREE.Clock();
    let frameId = 0;
    let yaw = 0;
    const orbitRadius = 34;
    const baseHeight = 16;
    const spinSpeed = 0.16;
    const bobSpeed = 0.28;
    const bobAmplitude = 0.65;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 0.05);
      const elapsed = clock.getElapsedTime();
      yaw += delta * spinSpeed;
      camera.position.x = Math.sin(yaw) * orbitRadius;
      camera.position.z = Math.cos(yaw) * orbitRadius;
      camera.position.y = baseHeight + Math.sin(elapsed * bobSpeed) * bobAmplitude;
      camera.lookAt(0, 6, 0);
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const nextWidth = mount.clientWidth || 320;
      const nextHeight = mount.clientHeight || 280;
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    };

    window.addEventListener("resize", handleResize);
    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      mount.innerHTML = "";
    };
  }, [events]);

  return <div ref={mountRef} className="three-scene" />;
}

function SpaceTimeLegend3D() {
  const items = [
    ["Batı Anadolu / Ege Graben", "#56b4e9"],
    ["Doğu Anadolu Fay", "#e69f00"],
    ["Kuzey Anadolu Fay", "#111827"],
    ["Kuzey Anadolu / Karadeniz Kuşağı", "#cc79a7"],
    ["İç Anadolu Sismik", "#009e73"],
    ["Akdeniz / Helenik-Kıbrıs Yay", "#d55e00"],
    ["Diğer", "#808080"],
  ];
  return (
    <div className="space-legend">
      <div className="time-axis-label">
        Alt = eski, üst = yeni
      </div>
      <div className="legend-row">
        {items.map(([label, color]) => (
          <span key={label} className="legend-chip">
            <i style={{ backgroundColor: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

function PredictionComparisonChart({ history, metric, title, suffix = "", emptyText = "İlk tahmin-gerçek eşleşmesi bekleniyor." }) {
  const valueFor = (match, side) => Number(match?.[side]?.[metric]);
  const waitCapMinutes = 120;
  const series = (history || [])
    .filter((match) => Number.isFinite(valueFor(match, "predicted")) && Number.isFinite(valueFor(match, "actual")))
    .slice(-12);
  const labels = series.map((match) => formatShortTime(match.actual.date));
  const predictedRaw = series.map((match) => valueFor(match, "predicted"));
  const actualRaw = series.map((match) => valueFor(match, "actual"));
  const dynamicWaitMax = (() => {
    if (metric !== "waitMinutes") return null;
    const combined = [...predictedRaw, ...actualRaw]
      .filter((value) => Number.isFinite(value) && value >= 0)
      .sort((a, b) => a - b);
    if (!combined.length) return waitCapMinutes;
    const percentileIndex = Math.max(0, Math.ceil(combined.length * 0.85) - 1);
    const baseline = combined[percentileIndex] || combined.at(-1) || 15;
    const padded = Math.ceil((baseline * 1.25) / 5) * 5;
    return Math.max(15, Math.min(waitCapMinutes, padded));
  })();
  const visibleWaitMax = dynamicWaitMax ?? waitCapMinutes;
  const predicted = metric === "waitMinutes"
    ? predictedRaw.map((value) => Math.min(value, visibleWaitMax))
    : predictedRaw;
  const actual = metric === "waitMinutes"
    ? actualRaw.map((value) => Math.min(value, visibleWaitMax))
    : actualRaw;

  if (!series.length) {
    return (
      <div className="mini-chart empty-chart">
        <p>{title}</p>
        <small>{emptyText}</small>
      </div>
    );
  }

  return (
    <div className="mini-chart">
      <h3>{title}</h3>
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: "Tahmin",
              data: predicted,
              rawData: predictedRaw,
              backgroundColor: "rgba(15, 118, 110, 0.78)",
              borderColor: "#0f766e",
              borderWidth: 1,
              borderRadius: 4,
              minBarLength: metric === "waitMinutes" ? 4 : 0,
            },
            {
              label: "Gerçek",
              data: actual,
              rawData: actualRaw,
              backgroundColor: "rgba(220, 38, 38, 0.74)",
              borderColor: "#dc2626",
              borderWidth: 1,
              borderRadius: 4,
              minBarLength: metric === "waitMinutes" ? 4 : 0,
            },
          ],
        }}
        options={{
          maintainAspectRatio: false,
          scales: {
            x: { grid: { display: false } },
            y: {
              beginAtZero: true,
              ...(metric === "waitMinutes" ? { max: visibleWaitMax } : {}),
              ...(metric === "waitMinutes"
                ? {
                    ticks: {
                      callback: (value) => (Number(value) === visibleWaitMax ? `${visibleWaitMax}+` : value),
                    },
                  }
                : {}),
            },
          },
          plugins: {
            legend: { position: "bottom", labels: { boxWidth: 12 } },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const rawValue = ctx.dataset.rawData?.[ctx.dataIndex];
                  return `${ctx.dataset.label}: ${Number(rawValue ?? ctx.raw).toFixed(1)}${suffix}`;
                },
              },
            },
          },
        }}
      />
    </div>
  );
}

function PredictionComparisonTable({ history }) {
  const rows = (history || [])
    .filter((match) => match?.predicted && match?.actual)
    .slice(-8)
    .reverse();

  if (!rows.length) {
    return <p className="empty">İlk tahmin-gerçek eşleşmesi bekleniyor.</p>;
  }

  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Zaman</th>
            <th>Bölge</th>
            <th>Büyüklük</th>
            <th>Derinlik</th>
            <th>Bekleme</th>
            <th>Konum hatası</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((match) => (
            <tr key={match.actual?.id || match.actual?.date}>
              {(() => {
                const predictedRegion = displayRegionForPrediction(match.predicted);
                const actualRegion = classifyRegion(match.actual) || "-";
                const regionMatched = predictedRegion === actualRegion && predictedRegion !== "-";
                const matchedRegionStyle = regionMatched ? { color: regionColorCss(predictedRegion), fontWeight: 600 } : undefined;
                return (
                  <>
              <td>{formatShortTime(match.actual?.date)}</td>
              <td>
                <div className="comparison-cell-pair">
                  <span style={matchedRegionStyle}><strong>T</strong> {predictedRegion}</span>
                  <span style={matchedRegionStyle}><strong>G</strong> {actualRegion}</span>
                </div>
              </td>
              <td>
                <div className="comparison-cell-pair">
                  <span><strong>T</strong> {Number.isFinite(match.predicted?.magnitude) ? `ML ${match.predicted.magnitude.toFixed(1)}` : "-"}</span>
                  <span><strong>G</strong> {Number.isFinite(match.actual?.magnitude) ? `ML ${match.actual.magnitude.toFixed(1)}` : "-"}</span>
                </div>
              </td>
              <td>
                <div className="comparison-cell-pair">
                  <span><strong>T</strong> {Number.isFinite(match.predicted?.depth) ? `${match.predicted.depth.toFixed(1)} km` : "-"}</span>
                  <span><strong>G</strong> {Number.isFinite(match.actual?.depth) ? `${match.actual.depth.toFixed(1)} km` : "-"}</span>
                </div>
              </td>
              <td>
                <div className="comparison-cell-pair">
                  <span><strong>T</strong> {Number.isFinite(match.predicted?.waitMinutes) ? formatMinutes(match.predicted.waitMinutes) : "-"}</span>
                  <span><strong>G</strong> {Number.isFinite(match.actual?.waitMinutes) ? formatMinutes(match.actual.waitMinutes) : "-"}</span>
                </div>
              </td>
              <td>{Number.isFinite(match.distanceKm) ? `${match.distanceKm.toFixed(1)} km` : "-"}</td>
                  </>
                );
              })()}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatMaybePercent(value) {
  return value == null || Number.isNaN(value) ? "-" : `%${Math.round(Number(value))}`;
}

function formatMaybeKm(value) {
  return value == null || Number.isNaN(value) ? "-" : `${Math.round(Number(value))} km`;
}

function RegionPerformanceTable({ rows }) {
  const items = (rows || []).filter((row) => row.count > 0);
  if (!items.length) {
    return <p className="empty">Bölgesel performans için eşleşme birikiyor.</p>;
  }
  return (
    <div className="region-table">
      <table>
        <thead>
          <tr>
            <th>Bölge</th>
            <th>Değerlendirilen olay</th>
            <th>Ort. konum hatası</th>
            <th>&lt;250 km</th>
            <th>Ort. bekleme hatası</th>
            <th>Ort. derinlik hatası</th>
            <th>Ort. büyüklük hatası</th>
          </tr>
        </thead>
        <tbody>
          {items.map((row) => (
            <tr key={row.region}>
              <td>{row.region}</td>
              <td>{row.count}</td>
              <td>{row.meanLocationErrorKm != null ? `${row.meanLocationErrorKm} km` : "-"}</td>
              <td>{row.under250Rate != null ? `%${row.under250Rate}` : "-"}</td>
              <td>{row.meanWaitErrorMinutes != null ? formatMinutes(row.meanWaitErrorMinutes) : "-"}</td>
              <td>{row.meanDepthErrorKm != null ? `${row.meanDepthErrorKm} km` : "-"}</td>
              <td>{row.meanMagnitudeError != null ? `ML ${row.meanMagnitudeError}` : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RegionTransitionsTable({ rows }) {
  const items = (rows || []).filter((row) => row.count > 0);
  if (!items.length) {
    return <p className="empty">Bölge geçişleri için eşleşme birikiyor.</p>;
  }
  return (
    <div className="region-table">
      <table>
        <thead>
          <tr>
            <th>Geçiş</th>
            <th>Örnek</th>
            <th>Ort. konum hatası</th>
            <th>Ort. bekleme hatası</th>
            <th>Ort. büyüklük hatası</th>
          </tr>
        </thead>
        <tbody>
          {items.map((row) => (
            <tr key={`${row.fromRegion}-${row.toRegion}`}>
              <td>{row.fromRegion} -&gt; {row.toRegion}</td>
              <td>{row.count}</td>
              <td>{row.meanLocationErrorKm != null ? `${row.meanLocationErrorKm} km` : "-"}</td>
              <td>{row.meanWaitErrorMinutes != null ? formatMinutes(row.meanWaitErrorMinutes) : "-"}</td>
              <td>{row.meanMagnitudeError != null ? `ML ${row.meanMagnitudeError}` : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BandPerformanceTable({ rows, label }) {
  const items = (rows || []).filter((row) => row.count > 0);
  if (!items.length) {
    return <p className="empty">{label} için yeterli eşleşme birikiyor.</p>;
  }
  return (
    <div className="region-table">
      <table>
        <thead>
          <tr>
            <th>Bant</th>
            <th>Örnek</th>
            <th>Ort. konum hatası</th>
            <th>&lt;250 km</th>
            <th>Bölge isabeti</th>
            <th>Ort. bekleme hatası</th>
          </tr>
        </thead>
        <tbody>
          {items.map((row) => (
            <tr key={`${label}-${row.band}`}>
              <td>{row.band}</td>
              <td>{row.count}</td>
              <td>{row.meanLocationErrorKm != null ? `${row.meanLocationErrorKm} km` : "-"}</td>
              <td>{row.under250Rate != null ? `%${row.under250Rate}` : "-"}</td>
              <td>{row.regionAccuracyRate != null ? `%${row.regionAccuracyRate}` : "-"}</td>
              <td>{row.meanWaitErrorMinutes != null ? formatMinutes(row.meanWaitErrorMinutes) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SpaceTimeSequences({ items }) {
  if (!items?.length) {
    return <p className="empty">Uzay-zaman olay sırası birikiyor.</p>;
  }
  return (
    <div className="insight-list">
      {items.map((item) => (
        <article className="insight-card" key={item.label}>
          <div className="insight-head">
            <strong>{item.label}</strong>
            <span>{item.count} olay</span>
          </div>
          <div className="insight-meta">
            <span>Son olay {item.latestTimeLabel}</span>
            <span>Pencere {item.label}</span>
          </div>
          <ol className="sequence-list">
            {item.items.map((event) => (
              <li className="sequence-item" key={`${item.label}-${event.id}-${event.order}`}>
                <div className="sequence-row">
                  <span className="sequence-order">{event.order}.</span>
                  <span className="sequence-time">{event.timeLabel}</span>
                  <span className="region-chip">
                    <i style={{ background: regionColorCss(event.region) }} />
                    {event.region}
                  </span>
                </div>
                <div className="sequence-detail">
                  <span>ML {event.magnitude}</span>
                  <span>{event.depth} km</span>
                  <span>{event.latitude}, {event.longitude}</span>
                  <span>{event.deltaMinutes != null ? `+${event.deltaMinutes} dk` : "başlangıç"}</span>
                </div>
              </li>
            ))}
          </ol>
        </article>
      ))}
    </div>
  );
}

const REGION_SHORT_CODES = {
  "Kuzey Anadolu Fay Zonu": "KAF",
  "Kuzey Anadolu / Karadeniz Kusağı": "KAK",
  "Kuzey Anadolu / Karadeniz Kusaği": "KAK",
  "Kuzey Anadolu / Karadeniz Kusağı": "KAK",
  "Kuzey Anadolu / Karadeniz Kusağı": "KAK",
  "Doğu Anadolu Fay Zonu": "DAF",
  "Batı Anadolu / Ege Graben Zonu": "EGE",
  "İç Anadolu Sismik Zonu": "İA",
  "Akdeniz / Helenik-Kıbrıs Yay Zonu": "AKD",
  "Diğer": "DGR",
};

function regionShortCode(region) {
  return REGION_SHORT_CODES[region] || region;
}

function RegionCombinationPatterns({ items }) {
  if (!items?.length) {
    return <p className="empty">Kombinasyon örüntüleri birikiyor.</p>;
  }
  return (
    <div className="combo-grid">
      {items.map((group) => (
        <article className="combo-card" key={group.length}>
          <div className="combo-head">
            <strong>{group.length}'lü</strong>
            <span>{group.totalWindows} pencere</span>
          </div>
          {group.patterns?.length ? (
            <ol className="combo-list">
              {group.patterns.map((pattern, index) => (
                <li className="combo-item" key={`${group.length}-${index}-${pattern.sequence.join("|")}`}>
                  <div className="combo-sequence">
                    {pattern.sequence.map((region, regionIndex) => (
                      <span key={`${pattern.sequence.join("|")}-${regionIndex}`} className="combo-token">
                        <i style={{ background: regionColorCss(region) }} />
                        {regionShortCode(region)}
                      </span>
                    ))}
                  </div>
                  <div className="combo-meta">
                    <span>{pattern.count} kez</span>
                    <span>Son: {pattern.lastSeenLabel || "-"}</span>
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <p className="combo-empty">Bu uzunlukta tekrar eden dizilim henüz yok.</p>
          )}
        </article>
      ))}
    </div>
  );
}

const REGION_WINDOW_OPTIONS = [
  { value: "12", label: "12" },
  { value: "60", label: "60" },
  { value: "all", label: "Tümü" },
];

const SPACE_TIME_WINDOW_OPTIONS = [
  { value: "1", label: "1s" },
  { value: "3", label: "3s" },
  { value: "6", label: "6s" },
  { value: "12", label: "12s" },
];

function filterSpaceTimeEvents(events, windowValue) {
  const items = Array.isArray(events) ? events : [];
  if (!items.length) return [];
  const hours = Number(windowValue);
  if (!Number.isFinite(hours) || hours <= 0) return items;
  const latestTimestamp = items[0]?.timestamp;
  if (!Number.isFinite(latestTimestamp)) return items;
  const cutoff = latestTimestamp - (hours * 3_600_000);
  return items.filter((event) => Number.isFinite(event.timestamp) && event.timestamp >= cutoff);
}

function App() {
  const liveState = useLiveState();
  const statusState = useTrainingStatus();
  const state = liveState.data;
  const [trainingExisting, setTrainingExisting] = useState(false);
  const [checkingAfad, setCheckingAfad] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resettingEvaluation, setResettingEvaluation] = useState(false);
  const [resettingConfusion, setResettingConfusion] = useState(false);
  const [refreshingCombos, setRefreshingCombos] = useState(false);
  const [trainMessage, setTrainMessage] = useState("");
  const [toastMessage, setToastMessage] = useState("");
  const [now, setNow] = useState(Date.now());
  const [regionWindow, setRegionWindow] = useState("60");
  const [spaceTimeWindow, setSpaceTimeWindow] = useState("12");
  const previousStatusRef = useRef(null);
  const previousTrainingRunCountRef = useRef(null);
  const latestEventsSource = state?.latestEvents || [];
  const filteredSpaceTimeEvents = useMemo(
    () => filterSpaceTimeEvents(latestEventsSource, spaceTimeWindow),
    [latestEventsSource, spaceTimeWindow],
  );

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const metrics = state?.metrics || {};
  const fallbackMetrics = statusState?.data?.metrics || {};
  const connectionIssue = liveState.error && statusState.error;
  const latestEvents = state?.latestEvents || [];
  const predictions = state?.predictions || [];
  const recentMatches = state?.recentMatches || [];
  const comparisonHistory = state?.comparisonHistory || [];
  const showMapHistory = state?.showMapHistory;
  const lastMatch = state?.lastMatch || null;
  const config = state?.config || {};
  const spaceTimeSequences = state?.spaceTimeSequences || [];
  const regionCombinationPatterns = state?.regionCombinationPatterns || [];
  const prediction = predictions[0];
  const recentPerformance = metrics.recentPerformance || {};
  const largeEventPerformance = metrics.largeEventPerformance || {};
  const helperModelReady = (metrics.totalTrainingRunCount ?? 0) > 0 && metrics.modelMemory !== "new";
  const largeEventSignalSummary = `${largeEventPerformance.truePositive ?? 0} doğru, ${largeEventPerformance.falsePositive ?? 0} yanlış, ${largeEventPerformance.pendingCount ?? 0} beklemede`;
  const largeEventPrecisionRecall = `${largeEventPerformance.precision != null ? `%${Math.round(largeEventPerformance.precision * 100)}` : "-"} / ${largeEventPerformance.recall != null ? `%${Math.round(largeEventPerformance.recall * 100)}` : "-"}`;
  const modelMemoryLabel = metrics.modelMemory === "loaded"
    ? "Kayıtlı model yüklendi"
    : metrics.modelMemory === "saved"
      ? "Model bu oturumda kaydedildi"
      : "Yeni model belleği";
  const modelMemoryDetail = metrics.modelMemory === "loaded"
    ? `Kaydedildi ${formatTime(metrics.modelSavedAt)}, yüklendi ${formatTime(metrics.modelLoadedAt)}`
    : metrics.modelMemory === "saved"
      ? `Bu oturumda kaydedildi ${formatTime(metrics.modelSavedAt)}`
      : "Henüz kayıtlı ağırlık yüklenmedi";
  const regionPerformanceRows = metrics.regionPerformanceByWindow?.[regionWindow] || metrics.regionPerformance || [];
  const magnitudePerformanceRows = metrics.magnitudePerformanceByWindow?.[regionWindow] || [];
  const depthPerformanceRows = metrics.depthPerformanceByWindow?.[regionWindow] || [];
  const regionTransitionRows = metrics.regionTransitionsByWindow?.[regionWindow] || metrics.regionTransitions || [];
  const liveDeltaMs = Math.max(0, now - (state?.receivedAt || now));
  const liveTotalLearningMs = metrics.totalLearningMs != null ? metrics.totalLearningMs + liveDeltaMs : (metrics.sessionElapsedMs || 0) + liveDeltaMs;
  const liveSessionMs = metrics.sessionElapsedMs != null ? metrics.sessionElapsedMs + liveDeltaMs : null;
  const activityStatusLabel = connectionIssue
    ? "backend_offline"
    : (metrics.error ? "error" : metrics.status) || (fallbackMetrics.error ? "error" : fallbackMetrics.status) || "idle";
  const trainingProgress = metrics.trainingProgress || fallbackMetrics.trainingProgress || {};
  const trainingProgressLines = trainingProgress?.active
    ? [
        trainingProgress.phaseLabel || null,
        trainingProgress.overallEpochs
          ? `Epoch ${trainingProgress.overallEpoch || 0}/${trainingProgress.overallEpochs}`
          : null,
        trainingProgress.totalBatches
          ? `Batch ${trainingProgress.currentBatch || 0}/${trainingProgress.totalBatches}`
          : null,
        trainingProgress.etaMs != null
          ? `Kalan süre yaklaşık ${formatDuration(trainingProgress.etaMs)}`
          : null,
      ].filter(Boolean)
    : [];
  const activityStatusDetailText = connectionIssue
    ? "Backend'e şu an ulaşılamıyor. API yeniden ayağa kalkınca pano otomatik toparlanacak."
    : metrics.error || metrics.lastAction || fallbackMetrics.error || fallbackMetrics.lastAction || "Bekliyor";
  const activityStatusDetail = trainingProgressLines.length
    ? (
        <>
          <div>{activityStatusDetailText}</div>
          {trainingProgressLines.map((line) => <div key={line}>{line}</div>)}
        </>
      )
    : activityStatusDetailText;
  const loadingStatusDetail = trainingProgressLines.length
    ? (
        <>
          <div>{activityStatusLabel} · {activityStatusDetailText}</div>
          {trainingProgressLines.map((line) => <div key={`loading-${line}`}>{line}</div>)}
        </>
      )
    : `${activityStatusLabel} · ${activityStatusDetailText}`;
  const initialTrainingRequired = Boolean(metrics.initialTrainingRequired ?? fallbackMetrics.initialTrainingRequired);

  useEffect(() => {
    if (!state) return;
    const previousStatus = previousStatusRef.current;
    const previousRunCount = previousTrainingRunCountRef.current;
    const currentRunCount = metrics.totalTrainingRunCount ?? 0;

    if (
      previousStatus === "training"
      && metrics.status !== "training"
      && currentRunCount > (previousRunCount ?? 0)
    ) {
      const lastAction = metrics.lastAction || "";
      const message = /tam eğitim/i.test(lastAction)
        ? "2 yıllık ana eğitim tamamlandı."
        : /hafif eğitim/i.test(lastAction)
          ? "Canlı hafif eğitim tamamlandı."
          : initialTrainingRequired
            ? "Geçmiş katalog eğitimi tamamlandı."
            : "Arka plan eğitimi tamamlandı.";
      setToastMessage(message);
      setTrainMessage(metrics.lastAction || message);
    }

    previousStatusRef.current = metrics.status;
    previousTrainingRunCountRef.current = currentRunCount;
  }, [state, initialTrainingRequired, metrics.lastAction, metrics.status, metrics.totalTrainingRunCount]);

  useEffect(() => {
    if (!toastMessage) return undefined;
    const timer = setTimeout(() => setToastMessage(""), 5000);
    return () => clearTimeout(timer);
  }, [toastMessage]);

  if (!state) {
    return (
      <main className="loading">
        <Radar />
        <div>
          <div>{connectionIssue ? "Canlı pano backend bekliyor..." : "Canlı pano hazırlanıyor..."}</div>
          <small>{loadingStatusDetail}</small>
        </div>
      </main>
    );
  }

  async function trainExistingNow() {
    setTrainingExisting(true);
    try {
      const response = await fetch("/api/train-existing", { method: "POST" });
      const payload = await response.json();
      setTrainMessage(payload.metrics?.lastAction || payload.trainResult?.reason || "");
    } catch {
      setTrainMessage("Mevcut verilerle eğitim sırasında bağlantı sorunu oluştu.");
    } finally {
      setTrainingExisting(false);
    }
  }

  async function trainNow() {
    setCheckingAfad(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000);
    try {
      const response = await fetch("/api/train", { method: "POST", signal: controller.signal });
      const payload = await response.json();
      setTrainMessage(payload.metrics?.lastAction || payload.trainResult?.reason || "");
    } catch (error) {
      setTrainMessage(error?.name === "AbortError"
        ? "AFAD kontrolü uzun sürdü; istek iptal edildi."
        : "Eğitim isteği sırasında bağlantı sorunu oluştu.");
    } finally {
      clearTimeout(timeout);
      setCheckingAfad(false);
    }
  }

  async function resetModelNow() {
    setResetting(true);
    try {
      const response = await fetch("/api/reset", { method: "POST" });
      const payload = await response.json();
      setTrainMessage(payload.metrics?.lastAction || "Model ve göstergeler sıfırlandı.");
    } catch (error) {
      setTrainMessage("Sıfırlama isteği sırasında bağlantı sorunu oluştu.");
    } finally {
      setResetting(false);
    }
  }

  async function resetEvaluationNow() {
    setResettingEvaluation(true);
    try {
      const response = await fetch("/api/evaluation/reset", { method: "POST" });
      const payload = await response.json();
      setTrainMessage(payload.metrics?.lastAction || "Değerlendirme tabloları sıfırlandı.");
    } catch {
      setTrainMessage("Değerlendirme tabloları sıfırlanırken bağlantı sorunu oluştu.");
    } finally {
      setResettingEvaluation(false);
    }
  }

  async function resetConfusionNow() {
    setResettingConfusion(true);
    try {
      const response = await fetch("/api/confusion/reset", { method: "POST" });
      const payload = await response.json();
      setTrainMessage(payload.metrics?.lastAction || "TP / FP / FN / TN göstergeleri sıfırlandı.");
    } catch {
      setTrainMessage("TP / FP / FN / TN göstergeleri sıfırlanırken bağlantı sorunu oluştu.");
    } finally {
      setResettingConfusion(false);
    }
  }

  async function clearMapHistory() {
    await fetch("/api/map/clear", { method: "POST" });
  }

    async function refreshRegionCombinationsNow() {
    setRefreshingCombos(true);
    try {
      const response = await fetch("/api/region-combinations/refresh", { method: "POST" });
      const payload = await response.json();
      setTrainMessage(payload.metrics?.lastRegionCombinationAt
        ? `Zon kombinasyonları güncellendi: ${formatTime(payload.metrics.lastRegionCombinationAt)}`
        : "Zon kombinasyonları güncellendi.");
    } catch {
      setTrainMessage("Zon kombinasyonları güncellenirken bağlantı sorunu oluştu.");
    } finally {
      setRefreshingCombos(false);
      }
    }

  return (
    <main>
      <header className="topbar">
        <div>
          <p className="eyebrow">AFAD canlı veri + çevrimiçi öğrenme</p>
          <h1>Deprem Tahmin Panosu</h1>
        </div>
          <div className="topbar-actions">
            <button className="ghost-button" onClick={resetModelNow} disabled={trainingExisting || checkingAfad || resetting} title="Modeli, göstergeleri ve eşleşme geçmişini sıfırla">
              <RotateCcw size={16} /> {resetting ? "Sıfırlanıyor" : "Modeli sıfırla"}
            </button>
            <button className="ghost-button" onClick={trainExistingNow} disabled={trainingExisting || checkingAfad || resetting} title="AFAD'a gitmeden eldeki katalog verileriyle eğit">
              <BrainCircuit size={16} /> {trainingExisting ? "Eğitiliyor" : "Mevcut verilerle öğren"}
            </button>
            <button onClick={trainNow} disabled={initialTrainingRequired || trainingExisting || checkingAfad || resetting} title={initialTrainingRequired ? "Önce mevcut katalogla ilk eğitim tamamlanmalı" : "AFAD'ı şimdi kontrol et, yeni veri varsa al ve eğit"}>
              <Play size={17} /> {checkingAfad ? "AFAD kontrol ediliyor" : "AFAD'ı kontrol et + öğren"}
            </button>
        </div>
      </header>
      {toastMessage ? <div className="toast-notice">{toastMessage}</div> : null}
      {trainMessage ? <p className="inline-status">{trainMessage}</p> : null}
      <p className="inline-status">
        {initialTrainingRequired
          ? "Canlı AFAD takibi beklemede; önce geçmiş katalogla ilk eğitim tamamlanacak."
          : `Canlı AFAD takibi açık; sistem ayrıca yaklaşık her ${Math.round(config.fetchIntervalMs / 1000)} saniyede bir otomatik kontrol yapar.`}
      </p>

        <section className="notice">
        <TriangleAlert size={20} />
        <p>Bu ekran araştırma ve takip amaçlıdır. Depremler deterministik biçimde tahmin edilemez; model çıktıları afet uyarısı veya güvenlik kararı olarak kullanılmamalıdır.</p>
        </section>
      <div className="stats-toolbar">
        <button
          className="ghost-button"
          onClick={resetConfusionNow}
          disabled={resettingConfusion}
          title="TP, FP, FN, TN, precision, recall ve specificity göstergelerini sıfırla"
        >
          <RotateCcw size={16} /> {resettingConfusion ? "Sıfırlanıyor" : "Confusion'u sıfırla"}
        </button>
      </div>

      <section className="stats">
        <PerformanceGauge score={recentPerformance.overallScore} />
        <Stat icon={Radar} label="Son eşleşme skoru" value={recentPerformance.latestScore != null ? `${recentPerformance.latestScore}/100` : "-"} detail="En son kapanan tahmin" />
        <Stat icon={DatabaseZap} label="Ana ham olay sayısı" value={metrics.eventCount} detail={`2 yıl · başlangıç ${config.startDate || "-"} · ${metrics.freshEvents} yeni olay son çekimde`} />
        <Stat icon={DatabaseZap} label="4+ ham olay sayısı" value={metrics.largeEventTrainingEventCount ?? 0} detail={`${config.largeEventCatalogYears || 5} yıl · başlangıç ${config.largeEventTrainingStartDate || "-"}`} />
        <Stat
          icon={TriangleAlert}
          label="24 saatte 4+ riski"
          value={helperModelReady && prediction?.largeEventRisk != null ? `%${Math.round(prediction.largeEventRisk * 100)}` : "-"}
          detail={
            helperModelReady
              ? (
                  <>
                    <div>{`Canlı 4+ sinyal durumu: ${largeEventSignalSummary}`}</div>
                    <div>{`Precision / Recall: ${largeEventPrecisionRecall}`}</div>
                  </>
                )
              : "4+ örüntü skoru, ilk başarılı eğitim tamamlandıktan sonra görünür."
          }
        />
        <Stat icon={BrainCircuit} label="Şu an ne yapıyor" value={activityStatusLabel} detail={activityStatusDetail} />
        <Stat
          icon={Activity}
          label="Son kayıp"
          value={formatLoss(metrics.lastLoss)}
          detail={`Konum ${formatLoss(metrics.lastLocationLoss)}, grid ${formatLoss(metrics.lastGridLoss)}, skaler ${formatLoss(metrics.lastScalarLoss)}, bekleme ${formatLoss(metrics.lastWaitLoss)}, bölge ${formatLoss(metrics.lastRegionLoss)}`}
        />
        <Stat icon={Activity} label="Eğitim süresi" value={formatDuration(liveTotalLearningMs)} detail={`Bu açılış ${formatDuration(liveSessionMs)}, son eğitim ${formatDuration(metrics.lastTrainingMs)}`} />
        <Stat icon={SatelliteDish} label="Son veri çekimi" value={formatTime(metrics.lastFetchAt)} detail={`Son kontrolde +${metrics.freshEvents ?? 0} olay · ${Math.round(config.fetchIntervalMs / 1000)} sn aralık`} />
        <Stat icon={Radar} label="Son 12 ort. konum hatası" value={recentPerformance.meanLocationErrorKm != null ? `${recentPerformance.meanLocationErrorKm} km` : "-"} detail={`${recentPerformance.recentWindow ?? 0} eşleşme üzerinden`} />
        <Stat icon={MapPin} label="250 km altı başarı" value={recentPerformance.under250Rate != null ? `%${recentPerformance.under250Rate}` : "-"} detail="Son 12 eşleşmede" />
        <Stat icon={MapPin} label="Bölge isabet oranı" value={recentPerformance.regionAccuracyRate != null ? `%${recentPerformance.regionAccuracyRate}` : "-"} detail="Son 12 eşleşmede" />
        <Stat icon={Radar} label="TP" value={recentPerformance.truePositive ?? "-"} detail={`Orta+/isabetli · ${recentPerformance.confusionTotal ?? 0} kayıt`} />
        <Stat icon={Radar} label="FP" value={recentPerformance.falsePositive ?? "-"} detail={`Orta+/kaçırdı · ${recentPerformance.confusionTotal ?? 0} kayıt`} />
        <Stat icon={Radar} label="FN" value={recentPerformance.falseNegative ?? "-"} detail={`Düşük güven/isabetli · ${recentPerformance.confusionTotal ?? 0} kayıt`} />
        <Stat icon={Radar} label="TN" value={recentPerformance.trueNegative ?? "-"} detail={`Düşük güven/kaçırdı · ${recentPerformance.confusionTotal ?? 0} kayıt`} />
        <Stat icon={Radar} label="Precision" value={recentPerformance.precision != null ? `%${Math.round(recentPerformance.precision * 100)}` : "-"} detail={`TP / (TP + FP) · Eşik ${recentPerformance.confidenceThreshold != null ? `%${Math.round(recentPerformance.confidenceThreshold * 100)}` : "-"}`} />
        <Stat icon={Radar} label="Recall" value={recentPerformance.recall != null ? `%${Math.round(recentPerformance.recall * 100)}` : "-"} detail={`TP / (TP + FN) · Eşik ${recentPerformance.confidenceThreshold != null ? `%${Math.round(recentPerformance.confidenceThreshold * 100)}` : "-"}`} />
        <Stat icon={Radar} label="Specificity" value={recentPerformance.specificity != null ? `%${Math.round(recentPerformance.specificity * 100)}` : "-"} detail={`TN / (TN + FP) · Eşik ${recentPerformance.confidenceThreshold != null ? `%${Math.round(recentPerformance.confidenceThreshold * 100)}` : "-"}`} />
        <Stat icon={Activity} label="Bekleme süresi hatası" value={formatMinutes(recentPerformance.meanWaitErrorMinutes)} detail="Son 12 eşleşme ortalaması" />
        <Stat icon={Activity} label="Büyüklük hatası" value={recentPerformance.meanMagnitudeError != null ? `ML ${recentPerformance.meanMagnitudeError}` : "-"} detail="Son 12 eşleşme ortalaması" />
        <Stat icon={Activity} label="Derinlik hatası" value={recentPerformance.meanDepthErrorKm != null ? `${recentPerformance.meanDepthErrorKm} km` : "-"} detail="Son 12 eşleşme ortalaması" />
            <Stat
              icon={BrainCircuit}
              label="Model"
              value={`GRU v${config.modelVersion}`}
              detail={`${config.lookback} olay, özet ${config.summaryWindows?.join("/")} · ${modelMemoryLabel}`}
            />
            <Stat
              icon={DatabaseZap}
              label="Model belleği"
              value={modelMemoryLabel}
              detail={modelMemoryDetail}
            />
      </section>

      <section className="hero-grid">
        <div className="panel map-panel">
          <div className="panel-head">
            <div>
              <h2>Konum İzleme</h2>
              <p>En son gerçek AFAD olayı ve modelin bir sonraki olay için ürettiği nokta.</p>
              <p className="panel-note">Haritadaki zon boyaları, backend tarafında olaylara atanan aynı sismotektonik bölge sınıflarını gösterir.</p>
            </div>
            <div className="panel-actions">
              <button className="ghost-button" onClick={clearMapHistory} title="Eski gerçek olay ve eski tahmini haritadan kaldır">
                <RotateCcw size={16} /> Haritayı temizle
              </button>
              <MapPin />
            </div>
          </div>
          <div className="chart tall">
            <MiniMap events={latestEvents} predictions={predictions} recentMatches={recentMatches} showMapHistory={showMapHistory} />
          </div>
        </div>

        <div className="panel comparison-panel">
          <h2>{"Tahmin-Ger\u00e7ek K\u0131yaslar\u0131"}</h2>
          <p className="panel-note">Son kapanan eşleşmelerde tahmin ile gerçek değerler yan yana.</p>
          <PredictionComparisonTable history={comparisonHistory} />
        </div>
      </section>

      <section className="support-grid">
        <div className="panel prediction">
          <h2>Sonraki Olay Tahmini</h2>
          {prediction ? (
            <>
              <strong>{prediction.latitude.toFixed(4)}, {prediction.longitude.toFixed(4)}</strong>
              <dl>
                <div><dt>Derinlik</dt><dd>{prediction.depth.toFixed(1)} km</dd></div>
                <div><dt>Tahmini büyüklük</dt><dd>ML {prediction.magnitude.toFixed(1)}</dd></div>
                <div><dt>Tahmini bekleme süresi</dt><dd>{formatMinutes(prediction.waitMinutes)}</dd></div>
                <div><dt>Tahmini bölge</dt><dd>{displayRegionForPrediction(prediction)}</dd></div>
                <div><dt>Bölge güveni</dt><dd>{prediction.regionConfidence != null ? `%${Math.round(prediction.regionConfidence * 100)}` : "-"}</dd></div>
                <div><dt>Tahmin alanı</dt><dd>{prediction.predictedMajorAxisKm != null && prediction.predictedMinorAxisKm != null ? `~${prediction.predictedMajorAxisKm} x ${prediction.predictedMinorAxisKm} km` : prediction.predictedRadiusKm != null ? `~${prediction.predictedRadiusKm} km` : "-"}</dd></div>
                <div><dt>Güven sınıfı</dt><dd>{prediction.confidenceClass || "-"}</dd></div>
                <div><dt>Eski tahmin - son gerçek</dt><dd>{lastMatch?.distanceKm?.toFixed?.(1) ?? "-"} km</dd></div>
                <div><dt>Güven sinyali</dt><dd>{Math.round(prediction.confidence * 100)}%</dd></div>
                <div><dt>Son referans</dt><dd>{formatTime(prediction.basedOnDate)}</dd></div>
              </dl>
            </>
          ) : (
            <p className="empty">
              {(metrics.eventCount || 0) > ((config.lookback || 16) + 1)
                ? "Aktif tahmin henüz hazırlanmadı; model belleği veya tahmin zinciri toparlanıyor."
                : `Tahmin için en az ${(config.lookback || 16) + 1} AFAD olayı bekleniyor.`}
            </p>
          )}
        </div>

        <div className="panel">
          <h2>{"Uzay-Zaman İzleme"}</h2>
          <p>{`${SPACE_TIME_WINDOW_OPTIONS.find((option) => option.value === spaceTimeWindow)?.label || "12s"} görünümündeki yıldız izi gibi mekansal akış.`}</p>
          <div className="chart trail-chart">
            <SpaceTimeTrail events={filteredSpaceTimeEvents} />
          </div>
        </div>
      </section>


      <section className="panel panel-3d-hero">
        <div className="panel-head compact">
          <div>
            <h2>3D Uzay-Zaman</h2>
            <p>{`Zaman yukarı doğru akar; yeni olaylar daha parlak görünür. Görünüm: ${SPACE_TIME_WINDOW_OPTIONS.find((option) => option.value === spaceTimeWindow)?.label || "12s"}`}</p>
          </div>
          <div className="segmented-control" role="tablist" aria-label="3D uzay-zaman penceresi">
            {SPACE_TIME_WINDOW_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={spaceTimeWindow === option.value ? "is-active" : ""}
                onClick={() => setSpaceTimeWindow(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <div className="space-time-grid">
          <div>
            <div className="chart chart-3d chart-3d-hero">
              <SpaceTimeScene3D events={filteredSpaceTimeEvents} />
            </div>
            <SpaceTimeLegend3D />
          </div>
          <div className="space-time-notes">
            <h3>Olay Sırası</h3>
            <p>3D akışın 1, 3, 6 ve 12 saatlik pencere içinde hangi zon sırasıyla ilerlediği listelenir.</p>
            <SpaceTimeSequences items={spaceTimeSequences} />
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head compact">
          <div>
            <h2>Zon Kombinasyonları</h2>
            <p>{`Kaydedilmiş deprem akışında tekrar eden 4’lü–11’li ardışık zon dizileri ve görülme sayıları. Son hesap: ${formatTime(metrics.lastRegionCombinationAt)}`}</p>
          </div>
          <button
            type="button"
            className="ghost-button"
            onClick={refreshRegionCombinationsNow}
            disabled={refreshingCombos}
            title="Zon kombinasyonlarını son verilere göre yeniden say"
          >
            <RotateCcw size={16} /> {refreshingCombos ? "Güncelleniyor" : "Şimdi yenile"}
          </button>
        </div>
        <RegionCombinationPatterns items={regionCombinationPatterns} />
      </section>

        <section className="table-wrap">
          <div className="panel-head compact">
            <div>
              <h2>Bölgesel performans</h2>
              <p>Sismotektonik zonlar için ayrı eşleşme metrikleri.</p>
            </div>
            <div className="panel-actions">
              <button
                type="button"
                className="ghost-button"
                onClick={resetEvaluationNow}
                disabled={resettingEvaluation}
                title="Bölgesel performans, geçiş, büyüklük ve derinlik tablolarını sıfırla"
              >
                <RotateCcw size={16} /> {resettingEvaluation ? "Sıfırlanıyor" : "Tabloları sıfırla"}
              </button>
              <div className="segmented-control" role="tablist" aria-label="Bölgesel performans penceresi">
                {REGION_WINDOW_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={regionWindow === option.value ? "is-active" : ""}
                    onClick={() => setRegionWindow(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <RegionPerformanceTable rows={regionPerformanceRows} />
        </section>

        <section className="table-wrap">
          <div className="panel-head compact">
            <div>
              <h2>Bölge geçişleri</h2>
              <p>Önceki sismik zondan sonraki gerçekleşen olayın zonuna en sık geçişler. Görünüm: {regionWindow === "all" ? "Tümü" : `Son ${regionWindow}`}</p>
            </div>
          </div>
          <RegionTransitionsTable rows={regionTransitionRows} />
        </section>

        <section className="table-wrap">
          <div className="panel-head compact">
            <div>
              <h2>Büyüklüğe göre tahmin kalitesi</h2>
              <p>Gerçek deprem büyüklük bandına göre konum, bölge ve bekleme başarısı. Görünüm: {regionWindow === "all" ? "Tümü" : `Son ${regionWindow}`}</p>
            </div>
          </div>
          <BandPerformanceTable rows={magnitudePerformanceRows} label="Büyüklük bantları" />
        </section>

        <section className="table-wrap">
          <div className="panel-head compact">
            <div>
              <h2>Derinliğe göre tahmin kalitesi</h2>
              <p>Gerçek deprem derinlik bandına göre konum, bölge ve bekleme başarısı. Görünüm: {regionWindow === "all" ? "Tümü" : `Son ${regionWindow}`}</p>
            </div>
          </div>
          <BandPerformanceTable rows={depthPerformanceRows} label="Derinlik bantları" />
        </section>

      <section className="table-wrap">
        <div className="panel-head compact">
          <div>
            <h2>Son AFAD Kayıtları</h2>
            <p>Öğrenme başlangıcı: {config.startDate}</p>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Zaman</th>
              <th>Yer</th>
              <th>ML</th>
              <th>Derinlik</th>
              <th>Koordinat</th>
            </tr>
          </thead>
          <tbody>
            {latestEvents.slice(0, 18).map((event) => (
              <tr key={event.id}>
                <td>{formatTime(event.date)}</td>
                <td>{event.location}</td>
                <td>{event.magnitude.toFixed(1)}</td>
                <td>{event.depth.toFixed(1)} km</td>
                <td>{event.latitude.toFixed(3)}, {event.longitude.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);




