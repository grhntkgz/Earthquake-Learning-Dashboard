export const REGION_ORDER = [
  "Kuzey Anadolu Fay Zonu",
  "Kuzey Anadolu / Karadeniz Kusa\u011fi",
  "Do\u011fu Anadolu Fay Zonu",
  "Bat\u0131 Anadolu / Ege Graben Zonu",
  "\u0130\u00e7 Anadolu Sismik Zonu",
  "Akdeniz / Helenik-K\u0131br\u0131s Yay Zonu",
  "Di\u011fer",
];

export const REGION_COLORS = {
  "Kuzey Anadolu Fay Zonu": 0x111827,
  "Kuzey Anadolu / Karadeniz Kusa\u011fi": 0xcc79a7,
  "Do\u011fu Anadolu Fay Zonu": 0xe69f00,
  "Bat\u0131 Anadolu / Ege Graben Zonu": 0x56b4e9,
  "\u0130\u00e7 Anadolu Sismik Zonu": 0x009e73,
  "Akdeniz / Helenik-K\u0131br\u0131s Yay Zonu": 0xd55e00,
  "Di\u011fer": 0x808080,
};

export const REGION_ZONES = [
  {
    label: "Bat\u0131 Anadolu / Ege Graben Zonu",
    borderColor: "rgba(86, 180, 233, 0.08)",
    backgroundColor: "rgba(86, 180, 233, 0.24)",
    segments: [[
      { x: 24.5, y: 36.7 },
      { x: 31.4, y: 36.7 },
      { x: 31.4, y: 39.9 },
      { x: 24.5, y: 39.9 },
      { x: 24.5, y: 36.7 },
    ]],
  },
  {
    label: "\u0130\u00e7 Anadolu Sismik Zonu",
    borderColor: "rgba(0, 158, 115, 0.08)",
    backgroundColor: "rgba(0, 158, 115, 0.18)",
    segments: [[
      { x: 31.4, y: 36.7 },
      { x: 36.8, y: 36.7 },
      { x: 36.8, y: 39.9 },
      { x: 31.4, y: 39.9 },
      { x: 31.4, y: 36.7 },
    ]],
  },
  {
    label: "Do\u011fu Anadolu Fay Zonu",
    borderColor: "rgba(230, 159, 0, 0.08)",
    backgroundColor: "rgba(230, 159, 0, 0.18)",
    segments: [[
      { x: 36.8, y: 36.7 },
      { x: 46.0, y: 36.7 },
      { x: 46.0, y: 39.9 },
      { x: 36.8, y: 39.9 },
      { x: 36.8, y: 36.7 },
    ]],
  },
  {
    label: "Kuzey Anadolu Fay Zonu",
    borderColor: "rgba(17, 24, 39, 0.08)",
    backgroundColor: "rgba(17, 24, 39, 0.14)",
    segments: [[
      { x: 24.5, y: 39.9 },
      { x: 46.0, y: 39.9 },
      { x: 46.0, y: 42.3 },
      { x: 24.5, y: 42.3 },
      { x: 24.5, y: 39.9 },
    ]],
  },
  {
    label: "Kuzey Anadolu / Karadeniz Kusa\u011fi",
    borderColor: "rgba(204, 121, 167, 0.08)",
    backgroundColor: "rgba(204, 121, 167, 0.14)",
    segments: [[
      { x: 24.5, y: 42.3 },
      { x: 46.0, y: 42.3 },
      { x: 46.0, y: 43.8 },
      { x: 24.5, y: 43.8 },
      { x: 24.5, y: 42.3 },
    ]],
  },
  {
    label: "Akdeniz / Helenik-K\u0131br\u0131s Yay Zonu",
    borderColor: "rgba(213, 94, 0, 0.08)",
    backgroundColor: "rgba(213, 94, 0, 0.18)",
    segments: [[
      { x: 24.5, y: 34.0 },
      { x: 46.0, y: 34.0 },
      { x: 46.0, y: 36.7 },
      { x: 24.5, y: 36.7 },
      { x: 24.5, y: 34.0 },
    ]],
  },
];

export const REGION_LABELS = [
  { text: "Bat\u0131 Anadolu / Ege Graben", shortText: "Bat\u0131 Anadolu / Ege", x: 28.25, y: 38.45 },
  { text: "\u0130\u00e7 Anadolu Sismik", shortText: "\u0130\u00e7 Anadolu", x: 34.1, y: 38.2 },
  { text: "Do\u011fu Anadolu Fay", shortText: "Do\u011fu Anadolu", x: 40.3, y: 38.2 },
  { text: "Kuzey Anadolu Fay", shortText: "Kuzey Anadolu Fay", x: 35.9, y: 40.7 },
  { text: "Kuzey Anadolu / Karadeniz Kusa\u011fi", shortText: "Karadeniz Ku\u015fa\u011f\u0131", x: 35.9, y: 43.15 },
  { text: "Akdeniz / Helenik-K\u0131br\u0131s Yay", shortText: "Akdeniz / Helenik-K\u0131br\u0131s", x: 35.2, y: 35.0 },
];

export const REGION_BOUNDS = {
  "Bat\u0131 Anadolu / Ege Graben Zonu": { lonMin: 24.5, lonMax: 31.4, latMin: 36.7, latMax: 39.9 },
  "\u0130\u00e7 Anadolu Sismik Zonu": { lonMin: 31.4, lonMax: 36.8, latMin: 36.7, latMax: 39.9 },
  "Do\u011fu Anadolu Fay Zonu": { lonMin: 36.8, lonMax: 46.0, latMin: 36.7, latMax: 39.9 },
  "Kuzey Anadolu Fay Zonu": { lonMin: 24.5, lonMax: 46.0, latMin: 39.9, latMax: 42.3 },
  "Kuzey Anadolu / Karadeniz Kusa\u011fi": { lonMin: 24.5, lonMax: 46.0, latMin: 42.3, latMax: 43.8 },
  "Akdeniz / Helenik-K\u0131br\u0131s Yay Zonu": { lonMin: 24.5, lonMax: 46.0, latMin: 34.0, latMax: 36.7 },
  "Di\u011fer": { lonMin: 24.0, lonMax: 46.0, latMin: 34.0, latMax: 43.8 },
};

export function classifyRegion(event) {
  const lat = Number(event?.latitude);
  const lon = Number(event?.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return "Di\u011fer";

  if (lat >= 34.0 && lat < 36.7 && lon >= 24.5 && lon <= 46.0) {
    return "Akdeniz / Helenik-K\u0131br\u0131s Yay Zonu";
  }

  if (lat >= 42.3 && lat <= 43.8 && lon >= 24.5 && lon <= 46.0) {
    return "Kuzey Anadolu / Karadeniz Kusa\u011fi";
  }

  if (lat >= 39.9 && lat < 42.3 && lon >= 24.5 && lon <= 46.0) {
    return "Kuzey Anadolu Fay Zonu";
  }

  if (lat >= 36.7 && lat < 39.9 && lon >= 24.5 && lon < 31.4) {
    return "Bat\u0131 Anadolu / Ege Graben Zonu";
  }

  if (lat >= 36.7 && lat < 39.9 && lon >= 31.4 && lon < 36.8) {
    return "\u0130\u00e7 Anadolu Sismik Zonu";
  }

  if (lat >= 36.7 && lat < 39.9 && lon >= 36.8 && lon <= 46.0) {
    return "Do\u011fu Anadolu Fay Zonu";
  }

  return "Di\u011fer";
}
