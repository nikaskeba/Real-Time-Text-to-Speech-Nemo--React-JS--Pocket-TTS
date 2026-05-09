import { useEffect, useMemo, useRef, useState } from "react";

const getDefaultProxyWsUrl = () => {
  if (import.meta.env.VITE_PROXY_WS_URL) {
    return import.meta.env.VITE_PROXY_WS_URL;
  }
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws`;
  }
  return "ws://127.0.0.1:8787/ws";
};

const getDefaultProxyHttpUrl = () => {
  if (import.meta.env.VITE_PROXY_HTTP_URL) {
    return import.meta.env.VITE_PROXY_HTTP_URL;
  }
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  return "http://127.0.0.1:8787";
};

const LOCAL_PROXY_URL = getDefaultProxyWsUrl();
const LOCAL_PROXY_HTTP_URL = getDefaultProxyHttpUrl();
const DEFAULT_SAMPLE_RATE = Number(import.meta.env.VITE_AUDIO_SAMPLE_RATE ?? 16000);
const DEFAULT_VISIBLE_WORDS = Number(import.meta.env.VITE_VISIBLE_TRANSCRIPT_WORDS ?? 60);
const FINALIZE_GRACE_MS = Number(import.meta.env.VITE_FINALIZE_GRACE_MS ?? 5000);
const DEFAULT_CHAT_MODEL = import.meta.env.VITE_LLM_MODEL ?? "gemma-3-4b-it";
const DEFAULT_CHAT_MAX_TOKENS = import.meta.env.VITE_LLM_MAX_TOKENS ?? "96";
const DEFAULT_CHAT_CONTEXT_WINDOW = import.meta.env.VITE_LLM_CONTEXT_WINDOW ?? "2048";
const DEFAULT_CHAT_SYSTEM_PROMPT =
  "You are a concise voice assistant. Reply naturally in one or two short spoken paragraphs.";
const DEFAULT_CHAT_ROOM_TONE_ENABLED =
  import.meta.env.VITE_CHAT_ROOM_TONE_ENABLED !== "0";
const DEFAULT_CHAT_ROOM_TONE_VOLUME = import.meta.env.VITE_CHAT_ROOM_TONE_VOLUME ?? "2.5";
const MAX_ROOM_TONE_GAIN = 0.04;
const DUPLICATE_CHAT_TURN_WINDOW_MS = 60000;
const buildChatAsrVadConfig = (mode = "balanced") => ({
  type: "configure",
  mode: "asr_vad",
  emit_vad_frames: true,
  auto_finalize_on_vad_end: true,
  min_silence_seconds: mode === "fast" ? 0.45 : 0.7,
});

const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
];

const getSupportedMimeType = () => {
  if (!window.MediaRecorder) {
    return "";
  }
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
};

const getAudioDeviceLabel = (device, index) =>
  device.label || `Microphone ${index + 1}`;

const formatTime = (date = new Date()) =>
  date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

const WINDOWS_1252_BYTE_MAP = new Map([
  [0x20AC, 0x80],
  [0x201A, 0x82],
  [0x0192, 0x83],
  [0x201E, 0x84],
  [0x2026, 0x85],
  [0x2020, 0x86],
  [0x2021, 0x87],
  [0x02C6, 0x88],
  [0x2030, 0x89],
  [0x0160, 0x8A],
  [0x2039, 0x8B],
  [0x0152, 0x8C],
  [0x017D, 0x8E],
  [0x2018, 0x91],
  [0x2019, 0x92],
  [0x201C, 0x93],
  [0x201D, 0x94],
  [0x2022, 0x95],
  [0x2013, 0x96],
  [0x2014, 0x97],
  [0x02DC, 0x98],
  [0x2122, 0x99],
  [0x0161, 0x9A],
  [0x203A, 0x9B],
  [0x0153, 0x9C],
  [0x017E, 0x9E],
  [0x0178, 0x9F],
]);

const ENCODING_ARTIFACT_PATTERN = /[ÂÃâ�\u00F0\u0178\u009F\u02DC\u0160]/;

const countEncodingArtifacts = (text) =>
  (text.match(/[ÂÃâ�\u00F0\u0178\u009F\u02DC\u0160]/g) ?? []).length;

const normalizeSpeechText = (text) =>
  text
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/\u2026/g, "...")
    .replace(/\u00A0/g, " ");

const toWindows1252Bytes = (text) =>
  new Uint8Array(
    Array.from(text, (character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return WINDOWS_1252_BYTE_MAP.get(codePoint) ?? codePoint;
    }),
  );

const removeResidualEncodingArtifacts = (text) =>
  text
    .replace(/\u00F0[\u0178\u009F\u02DC\u0160\u0080-\u00BF]*/g, "")
    .replace(/[ÂÃâ�\u0178\u009F\u02DC\u0160]/g, "")
    .replace(/\s+/g, " ")
    .trim();

const repairTextEncoding = (value) => {
  if (typeof value !== "string" || !value) {
    return "";
  }

  const normalized = normalizeSpeechText(value);
  if (!ENCODING_ARTIFACT_PATTERN.test(normalized)) {
    return normalized;
  }

  try {
    const bytes = toWindows1252Bytes(normalized);
    const decoded = normalizeSpeechText(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    return countEncodingArtifacts(decoded) < countEncodingArtifacts(normalized)
      ? decoded
      : removeResidualEncodingArtifacts(normalized);
  } catch {
    return removeResidualEncodingArtifacts(
      normalized
      .replace(/â[\u0080\u0099\u0098\u02DC\u2122]+/g, "'")
      .replace(/â[\u0080\u009C\u009D]+/g, '"')
      .replace(/â[\u0080\u0093\u0094]+/g, "-")
      .replace(/â[\u0080\u00A6]+/g, "...")
      .replace(/Â/g, ""),
    );
  }
};

const readTranscript = (payload) => {
  if (typeof payload === "string") {
    try {
      const parsed = JSON.parse(payload);
      const nestedText =
        parsed.channel?.alternatives?.[0]?.transcript ??
        parsed.results?.channels?.[0]?.alternatives?.[0]?.transcript ??
        parsed.alternatives?.[0]?.transcript;
      const explicitText =
        parsed.text ?? parsed.transcript ?? parsed.result ?? nestedText ?? parsed.message;

      return {
        parsed,
        text: repairTextEncoding(explicitText ?? JSON.stringify(parsed)),
        isExplicitTranscript: explicitText != null,
      };
    } catch {
      return { parsed: null, text: repairTextEncoding(payload), isExplicitTranscript: true };
    }
  }

  return { parsed: null, text: "", isExplicitTranscript: false };
};

const asTranscriptText = (segments) =>
  segments
    .map((segment, index) => {
      const speakerChanged = segment.speaker && segment.speaker !== segments[index - 1]?.speaker;
      return speakerChanged ? `${formatSpeaker(segment.speaker)}: ${segment.text}` : segment.text;
    })
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

const formatSpeaker = (speaker) => {
  if (!speaker) {
    return "";
  }

  const label = speaker
    .toString()
    .replace(/^speaker[_\s-]*/i, "Speaker ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return label || speaker.toString();
};

const readSpeaker = (parsed) =>
  parsed?.speaker ??
  parsed?.speaker_label ??
  parsed?.speakerLabel ??
  parsed?.speaker_id ??
  parsed?.speakerId ??
  parsed?.speaker_segment?.speaker ??
  parsed?.speakerSegment?.speaker ??
  parsed?.diarization?.speaker ??
  "";

const readSpeakerSegment = (parsed) =>
  parsed?.speaker_segment ?? parsed?.speakerSegment ?? parsed?.diarization_segment ?? null;

const toFiniteNumber = (value) => {
  if (value == null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const readStartMs = (value) => {
  const speakerSegment = readSpeakerSegment(value);
  const explicitMs =
    toFiniteNumber(value?.start_ms) ??
    toFiniteNumber(value?.startMs) ??
    toFiniteNumber(speakerSegment?.start_ms) ??
    toFiniteNumber(speakerSegment?.startMs);
  if (explicitMs != null) {
    return Math.max(0, Math.round(explicitMs));
  }

  const seconds = toFiniteNumber(value?.start) ?? toFiniteNumber(speakerSegment?.start);
  return seconds == null ? null : Math.max(0, Math.round(seconds * 1000));
};

const readEndMs = (value) => {
  const speakerSegment = readSpeakerSegment(value);
  const explicitMs =
    toFiniteNumber(value?.end_ms) ??
    toFiniteNumber(value?.endMs) ??
    toFiniteNumber(speakerSegment?.end_ms) ??
    toFiniteNumber(speakerSegment?.endMs);
  if (explicitMs != null) {
    return Math.max(0, Math.round(explicitMs));
  }

  const seconds = toFiniteNumber(value?.end) ?? toFiniteNumber(speakerSegment?.end);
  return seconds == null ? null : Math.max(0, Math.round(seconds * 1000));
};

const formatSubtitleTimestamp = (milliseconds) => {
  if (milliseconds == null) {
    return null;
  }

  const totalMilliseconds = Math.max(0, Math.round(milliseconds));
  const hours = Math.floor(totalMilliseconds / 3600000);
  const minutes = Math.floor((totalMilliseconds % 3600000) / 60000);
  const seconds = Math.floor((totalMilliseconds % 60000) / 1000);

  if (hours > 0) {
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  }

  return `${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}`;
};

const estimateCueDurationMs = (text) => {
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.min(12000, Math.max(1200, wordCount * 280));
};

const buildSubtitleCues = (segments, startedAt = null) => {
  const sessionStartedAtMs = startedAt ? Date.parse(startedAt) : null;
  const starts = segments.map((segment) => {
    const receivedAtMs = segment.receivedAt ? Date.parse(segment.receivedAt) : null;
    return (
      readStartMs(segment) ??
      (Number.isFinite(sessionStartedAtMs) && Number.isFinite(receivedAtMs)
        ? Math.max(0, receivedAtMs - sessionStartedAtMs)
        : null)
    );
  });

  return segments
    .map((segment, index) => {
      const text = repairTextEncoding(segment.text ?? "").replace(/\s+/g, " ").trim();
      const startMs = starts[index];
      const nextStartMs = starts[index + 1];
      const rawEndMs =
        readEndMs(segment) ??
        (nextStartMs != null ? nextStartMs : null) ??
        (startMs != null ? startMs + estimateCueDurationMs(text) : null);
      const endMs =
        startMs != null && rawEndMs != null && rawEndMs <= startMs
          ? startMs + estimateCueDurationMs(text)
          : rawEndMs;
      const speaker = segment.speaker ? formatSpeaker(segment.speaker) : "Speaker 1";

      return {
        speaker,
        start: formatSubtitleTimestamp(startMs),
        end: formatSubtitleTimestamp(endMs),
        text,
      };
    })
    .filter((cue) => cue.text);
};

const splitTranscriptIntoSubtitleSegments = (text, speaker = "Speaker 1") => {
  const cleaned = repairTextEncoding(text).replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return [];
  }

  const sentenceParts = cleaned
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter(Boolean);
  const sourceParts = sentenceParts.length ? sentenceParts : [cleaned];
  const chunks = [];

  for (const part of sourceParts) {
    const words = part.split(/\s+/).filter(Boolean);
    let current = [];

    for (const word of words) {
      current.push(word);
      const currentText = current.join(" ");
      const isNaturalBreak = /[,;:]$/.test(word);
      if (current.length >= 14 || currentText.length >= 86 || (isNaturalBreak && current.length >= 8)) {
        chunks.push(currentText);
        current = [];
      }
    }

    if (current.length) {
      chunks.push(current.join(" "));
    }
  }

  let cursorMs = 0;
  return chunks.map((chunk) => {
    const durationMs = estimateCueDurationMs(chunk);
    const segment = {
      text: chunk,
      speaker,
      startMs: cursorMs,
      endMs: cursorMs + durationMs,
    };
    cursorMs += durationMs + 250;
    return segment;
  });
};

const getNaturalWindow = (text, maxWords) => {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length || words.length <= maxWords) {
    return text;
  }

  const visibleWords = words.slice(-maxWords);
  const sentenceStartIndex = visibleWords.findIndex((word, index) => {
    if (index === 0) {
      return false;
    }
    return /[.!?]["')\]]?$/.test(visibleWords[index - 1]);
  });

  return (sentenceStartIndex > 0 ? visibleWords.slice(sentenceStartIndex) : visibleWords)
    .join(" ")
    .trim();
};

const isFinalTranscript = (parsed) => {
  if (!parsed) {
    return true;
  }

  if (typeof parsed.is_final === "boolean") {
    return parsed.is_final;
  }
  if (typeof parsed.final === "boolean") {
    return parsed.final;
  }
  if (typeof parsed.speech_final === "boolean") {
    return parsed.speech_final;
  }

  const type = (parsed.type ?? parsed.event ?? parsed.message_type ?? "")
    .toString()
    .toLowerCase();
  if (type.includes("partial") || type.includes("interim")) {
    return false;
  }

  return true;
};

const buildSessionFileName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `asr-session-${stamp}.json`;
};

const buildFileTranscriptionFileName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `asr-file-transcription-${stamp}.json`;
};

const buildTtsFileName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `tts-stream-${stamp}.json`;
};

const buildTtsAudioFileName = (extension = "wav") => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `tts-audio-${stamp}.${extension}`;
};

const buildVoiceChatFileName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `voice-chat-${stamp}.json`;
};

const downloadBlob = (blob, fileName) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
};

const downloadJson = (payload, fileName) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  downloadBlob(blob, fileName);
};

const base64ToBlob = (base64, mimeType = "audio/wav") => {
  const cleanBase64 = base64.replace(/^data:[^,]+,/, "").replace(/\s/g, "");
  const binary = atob(cleanBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mimeType });
};

const findAudioBase64 = (value, depth = 0) => {
  if (!value || depth > 4) {
    return "";
  }
  if (typeof value === "string") {
    return value.length > 80 ? value : "";
  }
  if (typeof value !== "object") {
    return "";
  }

  const preferredKeys = [
    "audioBase64",
    "audio_base64",
    "audio",
    "base64",
    "data",
    "wav",
  ];
  for (const key of preferredKeys) {
    const found = findAudioBase64(value[key], depth + 1);
    if (found) {
      return found;
    }
  }

  for (const child of Object.values(value)) {
    const found = findAudioBase64(child, depth + 1);
    if (found) {
      return found;
    }
  }
  return "";
};

const findMimeType = (value, depth = 0) => {
  if (!value || typeof value !== "object" || depth > 4) {
    return "";
  }

  const direct =
    value.mimeType ?? value.mime_type ?? value.contentType ?? value.content_type;
  if (typeof direct === "string" && direct) {
    return direct;
  }

  for (const child of Object.values(value)) {
    const found = findMimeType(child, depth + 1);
    if (found) {
      return found;
    }
  }
  return "";
};

const normalizeVoices = (payload) => {
  const result = payload?.result ?? payload;
  const data = result?.result ?? result?.data ?? result;
  const voices = data?.voices ?? data?.builtin_voices ?? (Array.isArray(data) ? data : []);
  const clones = data?.clones ?? data?.voice_clones ?? data?.cloned_voices ?? [];

  const systemVoices = (Array.isArray(voices) ? voices : []).map((voice) => {
    if (typeof voice === "string") {
      return {
        id: voice,
        label: voice,
        kind: "system",
      };
    }

    return {
      id: voice.id ?? voice.name ?? voice.voice ?? voice.slug,
      label: voice.label ?? voice.name ?? voice.voice ?? voice.id ?? "Voice",
      kind: "system",
    };
  });

  const clonedVoices = (Array.isArray(clones) ? clones : []).map((clone) => {
    if (typeof clone === "string") {
      return {
        id: clone,
        label: clone,
        kind: "clone",
      };
    }

    const id = clone.clone_id ?? clone.id ?? clone.name ?? clone.voice;
    const language = clone.language ? ` (${clone.language})` : "";
    return {
      id,
      label: `${id ?? "Clone"}${language}`,
      kind: "clone",
      language: clone.language,
      sourceFilename: clone.source_filename,
      createdAt: clone.created_at,
    };
  });

  return [...systemVoices, ...clonedVoices].filter((voice) => voice.id);
};

const normalizeLanguages = (payload) => {
  const result = payload?.result ?? payload;
  const data = result?.result ?? result?.data ?? result;
  const languages = data?.supported_languages ?? data?.languages ?? [];
  return Array.isArray(languages) ? languages : [];
};

const normalizeChatModels = (payload) => {
  const result = payload?.result ?? payload;
  const data = result?.result ?? result?.data ?? result?.models ?? result;
  const models = Array.isArray(data) ? data : [];
  const seen = new Set();

  return models
    .map((model) => {
      const id =
        typeof model === "string"
          ? model
          : model.id ?? model.name ?? model.model ?? model.slug;
      return id ? id.toString() : "";
    })
    .filter((id) => {
      if (!id || seen.has(id)) {
        return false;
      }
      seen.add(id);
      return true;
    });
};

const findTranscriptSegmentList = (value, depth = 0) => {
  if (!value || typeof value !== "object" || depth > 5) {
    return [];
  }

  if (Array.isArray(value)) {
    const segments = value
      .map((item) => {
        if (!item || typeof item !== "object") {
          return null;
        }
        const text =
          item.text ??
          item.transcript ??
          item.content ??
          item.words ??
          item.message ??
          "";
        const cleanedText = Array.isArray(text)
          ? text.map((word) => word.word ?? word.text ?? "").join(" ")
          : text.toString();
        const speaker = readSpeaker(item);
        if (!cleanedText.trim()) {
          return null;
        }

        return {
          text: repairTextEncoding(cleanedText).replace(/\s+/g, " ").trim(),
          speaker: speaker || "",
          speakerSegment: readSpeakerSegment(item),
          startMs: readStartMs(item),
          endMs: readEndMs(item),
        };
      })
      .filter(Boolean);

    return segments.length ? segments : [];
  }

  const preferredKeys = [
    "segments",
    "utterances",
    "diarization_segments",
    "speaker_segments",
    "results",
    "data",
    "result",
  ];

  for (const key of preferredKeys) {
    const found = findTranscriptSegmentList(value[key], depth + 1);
    if (found.length) {
      return found;
    }
  }

  for (const child of Object.values(value)) {
    const found = findTranscriptSegmentList(child, depth + 1);
    if (found.length) {
      return found;
    }
  }
  return [];
};

const asDiarizedTranscriptText = (segments) =>
  segments
    .map((segment, index) => {
      const speakerChanged = segment.speaker && segment.speaker !== segments[index - 1]?.speaker;
      return speakerChanged ? `${formatSpeaker(segment.speaker)}: ${segment.text}` : segment.text;
    })
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

const buildVadWebsocketUrl = (proxyUrl, mode = "balanced") => {
  try {
    const url = new URL(proxyUrl);
    url.searchParams.set("mode", "asr_vad");
    url.searchParams.set("emit_vad_frames", "1");
    url.searchParams.set("auto_finalize_on_vad_end", "1");
    url.searchParams.set("min_silence_seconds", mode === "fast" ? "0.45" : "0.7");
    return url.toString();
  } catch {
    const separator = proxyUrl.includes("?") ? "&" : "?";
    const silence = mode === "fast" ? "0.45" : "0.7";
    return `${proxyUrl}${separator}mode=asr_vad&emit_vad_frames=1&auto_finalize_on_vad_end=1&min_silence_seconds=${silence}`;
  }
};

const buildDiarizationOptions = (speakerCount) => {
  const count = Number(speakerCount);
  if (!Number.isFinite(count) || count <= 0) {
    return {};
  }

  const speakers = Math.floor(count);
  return {
    diarization_speakers: speakers,
    num_speakers: speakers,
    min_speakers: speakers,
    max_speakers: speakers,
  };
};

const buildDiarizedAsrWebsocketUrl = (proxyUrl, speakerCount) => {
  try {
    const url = new URL(proxyUrl);
    if (!url.searchParams.has("mode")) {
      url.searchParams.set("mode", "asr_vad");
    }
    url.searchParams.set("enable_diarization", "1");
    url.searchParams.set("emit_diarization_segments", "1");
    for (const [key, value] of Object.entries(buildDiarizationOptions(speakerCount))) {
      url.searchParams.set(key, value.toString());
    }
    return url.toString();
  } catch {
    const separator = proxyUrl.includes("?") ? "&" : "?";
    const hasMode = /[?&]mode=/.test(proxyUrl);
    const mode = hasMode ? "" : "mode=asr_vad&";
    const params = new URLSearchParams({
      enable_diarization: "1",
      emit_diarization_segments: "1",
      ...Object.fromEntries(
        Object.entries(buildDiarizationOptions(speakerCount)).map(([key, value]) => [
          key,
          value.toString(),
        ]),
      ),
    });
    return `${proxyUrl}${separator}${mode}${params.toString()}`;
  }
};

const extractChatDelta = (payload) =>
  payload?.choices?.[0]?.delta?.content ??
  payload?.choices?.[0]?.message?.content ??
  payload?.delta?.content ??
  payload?.content ??
  "";

const normalizeUtteranceKey = (text) =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .replace(/\s+/g, " ")
    .trim();

const isExpandedUtterance = (nextText, previousText) => {
  const next = normalizeUtteranceKey(nextText);
  const previous = normalizeUtteranceKey(previousText);
  return Boolean(previous && next !== previous && next.startsWith(`${previous} `));
};

const isDuplicateOrOlderUtterance = (nextText, previousText) => {
  const next = normalizeUtteranceKey(nextText);
  const previous = normalizeUtteranceKey(previousText);
  return Boolean(
    next &&
      previous &&
      (next === previous || previous.startsWith(`${next} `)),
  );
};

const formatLatency = (ms) => {
  if (ms == null) {
    return "";
  }
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
};

const createChatTiming = () => ({
  id: crypto.randomUUID(),
  speechStartedAt: null,
  speechEndedAt: null,
  transcriptReadyAt: null,
  chatRequestAt: null,
  firstLlmTokenAt: null,
  firstSpeakableAt: null,
  llmDoneAt: null,
  ttsRequestAt: null,
  firstTtsAudioAt: null,
  ttsDoneAt: null,
  transcriptSource: "",
  ttsPreview: "",
});

const durationBetween = (timing, fromKey, toKey) => {
  const from = timing?.[fromKey];
  const to = timing?.[toKey];
  if (from == null || to == null || to < from) {
    return null;
  }
  return to - from;
};

const buildChatTimingRows = (timing) => {
  if (!timing) {
    return [];
  }

  return [
    {
      id: "speech",
      label: "User speaking",
      detail: "mic speech start -> VAD speech end",
      valueMs: durationBetween(timing, "speechStartedAt", "speechEndedAt"),
    },
    {
      id: "asr",
      label: "ASR delay after silence",
      detail: timing.transcriptSource || "VAD speech end -> transcript usable",
      valueMs: durationBetween(timing, "speechEndedAt", "transcriptReadyAt"),
      priority: true,
    },
    {
      id: "llm-wait",
      label: "LLM wait",
      detail: "chat request -> first token",
      valueMs: durationBetween(timing, "chatRequestAt", "firstLlmTokenAt"),
      priority: true,
    },
    {
      id: "llm-stream",
      label: "LLM finishing",
      detail: "first token -> full response done",
      valueMs: durationBetween(timing, "firstLlmTokenAt", "llmDoneAt"),
      priority: true,
    },
    {
      id: "tts-wait",
      label: "TTS wait",
      detail: timing.ttsPreview || "tts request -> first audio chunk",
      valueMs: durationBetween(timing, "ttsRequestAt", "firstTtsAudioAt"),
      priority: true,
    },
    {
      id: "tts-playback",
      label: "TTS playback",
      detail: "first audio chunk -> playback drained",
      valueMs: durationBetween(timing, "firstTtsAudioAt", "ttsDoneAt"),
    },
    {
      id: "turn-pause",
      label: "Total pause",
      detail: "speech end -> assistant audio",
      valueMs: durationBetween(timing, "speechEndedAt", "firstTtsAudioAt"),
    },
  ];
};

const findSlowestChatStage = (rows) =>
  rows
    .filter((row) => row.priority && row.valueMs != null)
    .sort((left, right) => right.valueMs - left.valueMs)[0] ?? null;

const buildAlternatingChatMessages = (messages, limit = 12) => {
  const normalized = [];

  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") {
      continue;
    }

    const content = repairTextEncoding(message.text ?? message.content ?? "")
      .replace(/\s+/g, " ")
      .trim();
    if (!content) {
      continue;
    }

    const last = normalized.at(-1);
    if (last?.role === message.role) {
      last.content = `${last.content} ${content}`.trim();
    } else {
      normalized.push({
        role: message.role,
        content,
      });
    }
  }

  while (normalized.length && normalized[0].role !== "user") {
    normalized.shift();
  }

  let limited = normalized.slice(-limit);
  while (limited.length && limited[0].role !== "user") {
    limited = limited.slice(1);
  }

  if (limited.at(-1)?.role === "assistant") {
    limited = limited.slice(0, -1);
  }

  return limited;
};

const takeSpeakableText = (buffer, force = false, mode = "balanced") => {
  const normalized = buffer.replace(/\s+/g, " ").trimStart();
  if (!normalized.trim()) {
    return { text: "", remaining: "" };
  }

  if (force) {
    return { text: normalized.trim(), remaining: "" };
  }

  let sentenceEnd = -1;
  let phraseEnd = -1;
  for (let index = 0; index < normalized.length; index += 1) {
    const followedByBreak = index === normalized.length - 1 || /\s/.test(normalized[index + 1]);
    if (/[.!?]/.test(normalized[index]) && followedByBreak) {
      sentenceEnd = index + 1;
    }
    if (/[,;:]/.test(normalized[index]) && followedByBreak) {
      phraseEnd = index + 1;
    }
  }

  const sentenceThreshold = mode === "fast" ? 10 : 18;
  const phraseThreshold = mode === "fast" ? 18 : 36;
  const phraseWordThreshold = mode === "fast" ? 7 : 10;
  const fallbackWordThreshold = mode === "fast" ? 10 : 16;
  const fallbackSpeakWords = mode === "fast" ? 8 : 12;

  if (sentenceEnd >= sentenceThreshold) {
    return {
      text: normalized.slice(0, sentenceEnd).trim(),
      remaining: normalized.slice(sentenceEnd).trimStart(),
    };
  }

  const words = normalized.split(/\s+/);
  if (phraseEnd >= phraseThreshold || (phraseEnd >= 16 && words.length >= phraseWordThreshold)) {
    return {
      text: normalized.slice(0, phraseEnd).trim(),
      remaining: normalized.slice(phraseEnd).trimStart(),
    };
  }

  if (words.length >= fallbackWordThreshold) {
    return {
      text: words.slice(0, fallbackSpeakWords).join(" "),
      remaining: words.slice(fallbackSpeakWords).join(" "),
    };
  }

  return { text: "", remaining: buffer };
};

const readChatCompletionStream = async ({ response, signal, onDelta }) => {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Chat response body is not readable.");
  }

  const decoder = new TextDecoder();
  let pending = "";

  const processLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed || !trimmed.startsWith("data:")) {
      return false;
    }

    const data = trimmed.slice(5).trim();
    if (!data || data === "[DONE]") {
      return data === "[DONE]";
    }

    try {
      const payload = JSON.parse(data);
      const delta = extractChatDelta(payload);
      if (delta) {
        onDelta(delta, payload);
      }
    } catch {
      onDelta(data, null);
    }
    return false;
  };

  while (true) {
    if (signal?.aborted) {
      await reader.cancel();
      break;
    }

    const { done, value } = await reader.read();
    pending += decoder.decode(value ?? new Uint8Array(0), { stream: !done });
    const lines = pending.split(/\r?\n/);
    pending = lines.pop() ?? "";

    for (const line of lines) {
      if (processLine(line)) {
        await reader.cancel();
        return;
      }
    }

    if (done) {
      break;
    }
  }

  if (pending) {
    processLine(pending);
  }
};

const downsampleToInt16 = (input, inputSampleRate, outputSampleRate) => {
  if (outputSampleRate >= inputSampleRate) {
    return floatToInt16(input);
  }

  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Int16Array(outputLength);

  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(Math.floor((index + 1) * ratio), input.length);
    let sum = 0;

    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      sum += input[sampleIndex];
    }

    const average = sum / Math.max(end - start, 1);
    output[index] = floatSampleToInt16(average);
  }

  return output;
};

const floatToInt16 = (input) => {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = floatSampleToInt16(input[index]);
  }
  return output;
};

const floatSampleToInt16 = (sample) => {
  const clamped = Math.max(-1, Math.min(1, sample));
  return clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
};

const concatUint8Arrays = (chunks, totalLength) => {
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
};

const playPcmStream = async ({ response, onAudioContext, onChunk, signal }) => {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming response body is not readable.");
  }

  const AudioContext = window.AudioContext ?? window.webkitAudioContext;
  const audioContext = new AudioContext();
  onAudioContext?.(audioContext);
  await audioContext.resume?.();

  const sampleRate = Number(response.headers.get("x-audio-sample-rate") ?? 24000);
  let nextStartTime = audioContext.currentTime + 0.08;
  let leftover = new Uint8Array(0);
  let totalBytes = 0;
  let chunkCount = 0;

  const schedulePcm = (pcmBytes) => {
    if (pcmBytes.length < 2) {
      return;
    }

    const sampleCount = Math.floor(pcmBytes.length / 2);
    const audioBuffer = audioContext.createBuffer(1, sampleCount, sampleRate);
    const channel = audioBuffer.getChannelData(0);
    const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, sampleCount * 2);

    for (let index = 0; index < sampleCount; index += 1) {
      channel[index] = view.getInt16(index * 2, true) / 32768;
    }

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const startAt = Math.max(nextStartTime, audioContext.currentTime + 0.04);
    source.start(startAt);
    nextStartTime = startAt + audioBuffer.duration;
  };

  while (true) {
    if (signal?.aborted) {
      await reader.cancel();
      break;
    }

    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    const chunk = value ?? new Uint8Array(0);
    const combined = concatUint8Arrays([leftover, chunk], leftover.length + chunk.length);
    const playableLength = combined.length - (combined.length % 2);
    const playable = combined.slice(0, playableLength);
    leftover = combined.slice(playableLength);

    totalBytes += chunk.length;
    chunkCount += 1;
    onChunk?.({ chunkCount, totalBytes, sampleRate });
    schedulePcm(playable);
  }

  return {
    audioContext,
    chunkCount,
    totalBytes,
    sampleRate,
    estimatedDurationMs: Math.round((totalBytes / 2 / sampleRate) * 1000),
    playbackDelayMs: Math.max(0, Math.ceil((nextStartTime - audioContext.currentTime) * 1000)),
  };
};

const createRoomToneBed = async (volumePercent) => {
  const AudioContext = window.AudioContext ?? window.webkitAudioContext;
  if (!AudioContext) {
    throw new Error("This browser does not support Web Audio.");
  }

  const audioContext = new AudioContext();
  await audioContext.resume?.();

  const seconds = 2;
  const sampleCount = Math.floor(audioContext.sampleRate * seconds);
  const buffer = audioContext.createBuffer(1, sampleCount, audioContext.sampleRate);
  const channel = buffer.getChannelData(0);
  let pink = 0;

  for (let index = 0; index < sampleCount; index += 1) {
    const white = Math.random() * 2 - 1;
    pink = pink * 0.985 + white * 0.015;
    channel[index] = pink + white * 0.025;
  }

  const source = audioContext.createBufferSource();
  const highpass = audioContext.createBiquadFilter();
  const lowpass = audioContext.createBiquadFilter();
  const gain = audioContext.createGain();

  source.buffer = buffer;
  source.loop = true;
  highpass.type = "highpass";
  highpass.frequency.value = 90;
  lowpass.type = "lowpass";
  lowpass.frequency.value = 6200;
  gain.gain.value = 0;

  source.connect(highpass);
  highpass.connect(lowpass);
  lowpass.connect(gain);
  gain.connect(audioContext.destination);

  const setVolume = (nextVolumePercent) => {
    const normalized = Math.max(0, Math.min(10, Number(nextVolumePercent) || 0)) / 10;
    const targetGain = normalized * MAX_ROOM_TONE_GAIN;
    gain.gain.cancelScheduledValues(audioContext.currentTime);
    gain.gain.setTargetAtTime(targetGain, audioContext.currentTime, 0.08);
  };

  source.start();
  setVolume(volumePercent);

  return {
    context: audioContext,
    setVolume,
    stop: async (immediate = false) => {
      gain.gain.cancelScheduledValues(audioContext.currentTime);
      if (immediate) {
        gain.gain.value = 0;
      } else {
        gain.gain.setTargetAtTime(0, audioContext.currentTime, 0.12);
        await new Promise((resolve) => setTimeout(resolve, 350));
      }
      try {
        source.stop();
      } catch {
        // The source may already be stopped during teardown.
      }
      await audioContext.close?.();
    },
  };
};

const startPcmStream = ({ audioStream, socket, sampleRate, onChunk }) => {
  const AudioContext = window.AudioContext ?? window.webkitAudioContext;
  const audioContext = new AudioContext();
  audioContext.resume?.().catch(() => {});
  const source = audioContext.createMediaStreamSource(audioStream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silentGain = audioContext.createGain();

  silentGain.gain.value = 0;

  processor.onaudioprocess = (event) => {
    if (socket.readyState !== WebSocket.OPEN) {
      return;
    }

    const input = event.inputBuffer.getChannelData(0);
    const pcm = downsampleToInt16(input, audioContext.sampleRate, sampleRate);
    onChunk(pcm.byteLength);
    socket.send(pcm.buffer);
  };

  source.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(audioContext.destination);

  return {
    stop: () => {
      processor.disconnect();
      source.disconnect();
      silentGain.disconnect();
      audioContext.close();
    },
  };
};

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const [proxyUrl, setProxyUrl] = useState(LOCAL_PROXY_URL);
  const [proxyHttpUrl] = useState(LOCAL_PROXY_HTTP_URL);
  const [audioInputs, setAudioInputs] = useState([]);
  const [selectedAudioInputId, setSelectedAudioInputId] = useState("");
  const [audioInputType, setAudioInputType] = useState("microphone");
  const [audioMode, setAudioMode] = useState("pcm");
  const [diarizationSpeakerCount, setDiarizationSpeakerCount] = useState("2");
  const [sampleRate, setSampleRate] = useState(DEFAULT_SAMPLE_RATE);
  const [visibleWords, setVisibleWords] = useState(DEFAULT_VISIBLE_WORDS);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isFinishing, setIsFinishing] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [events, setEvents] = useState([]);
  const [transcriptSegments, setTranscriptSegments] = useState([]);
  const [incomingCount, setIncomingCount] = useState(0);
  const [lastIncoming, setLastIncoming] = useState("");
  const [sessionStartedAt, setSessionStartedAt] = useState(null);
  const [sessionEndedAt, setSessionEndedAt] = useState(null);
  const [error, setError] = useState("");
  const [fileInputMode, setFileInputMode] = useState("url");
  const [fileAudioUrl, setFileAudioUrl] = useState("");
  const [fileAudioFile, setFileAudioFile] = useState(null);
  const [fileResult, setFileResult] = useState(null);
  const [fileError, setFileError] = useState("");
  const [fileStatus, setFileStatus] = useState("Idle");
  const [isFileTranscribing, setIsFileTranscribing] = useState(false);
  const [ttsText, setTtsText] = useState("Hello from Pocket TTS");
  const [ttsVoice, setTtsVoice] = useState("alba");
  const [ttsLanguage, setTtsLanguage] = useState("english");
  const [ttsStreamFormat, setTtsStreamFormat] = useState("pcm_s16le");
  const [ttsVoices, setTtsVoices] = useState([]);
  const [ttsLanguages, setTtsLanguages] = useState([]);
  const [ttsResult, setTtsResult] = useState(null);
  const [ttsAudioSrc, setTtsAudioSrc] = useState("");
  const [ttsStatus, setTtsStatus] = useState("Idle");
  const [ttsError, setTtsError] = useState("");
  const [isTtsSynthesizing, setIsTtsSynthesizing] = useState(false);
  const [isTtsPlaying, setIsTtsPlaying] = useState(false);
  const [isTtsDownloading, setIsTtsDownloading] = useState(false);
  const [chatModel, setChatModel] = useState(DEFAULT_CHAT_MODEL);
  const [chatModels, setChatModels] = useState([]);
  const [chatModelsError, setChatModelsError] = useState("");
  const [isChatModelsLoading, setIsChatModelsLoading] = useState(false);
  const [chatMaxTokens, setChatMaxTokens] = useState(DEFAULT_CHAT_MAX_TOKENS);
  const [chatContextWindow, setChatContextWindow] = useState(DEFAULT_CHAT_CONTEXT_WINDOW);
  const [chatTurnMode, setChatTurnMode] = useState("fast");
  const [chatSystemPrompt, setChatSystemPrompt] = useState(DEFAULT_CHAT_SYSTEM_PROMPT);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatLatencyEvents, setChatLatencyEvents] = useState([]);
  const [chatTiming, setChatTiming] = useState(null);
  const [chatPartialTranscript, setChatPartialTranscript] = useState("");
  const [chatStatus, setChatStatus] = useState("Idle");
  const [chatVadStatus, setChatVadStatus] = useState("Waiting");
  const [chatError, setChatError] = useState("");
  const [isVoiceChatActive, setIsVoiceChatActive] = useState(false);
  const [isChatResponding, setIsChatResponding] = useState(false);
  const [isChatSpeaking, setIsChatSpeaking] = useState(false);
  const [isChatRoomToneEnabled, setIsChatRoomToneEnabled] = useState(
    DEFAULT_CHAT_ROOM_TONE_ENABLED,
  );
  const [chatRoomToneVolume, setChatRoomToneVolume] = useState(
    DEFAULT_CHAT_ROOM_TONE_VOLUME,
  );
  const [cloneId, setCloneId] = useState("");
  const [cloneLanguage, setCloneLanguage] = useState("english");
  const [cloneFile, setCloneFile] = useState(null);
  const [deleteCloneId, setDeleteCloneId] = useState("");
  const [cloneStatus, setCloneStatus] = useState("Idle");
  const [cloneError, setCloneError] = useState("");
  const [cloneResult, setCloneResult] = useState(null);
  const [isCloningVoice, setIsCloningVoice] = useState(false);
  const [isDeletingClone, setIsDeletingClone] = useState(false);

  const socketRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const finalizeTimerRef = useRef(null);
  const ttsAbortRef = useRef(null);
  const ttsAudioContextRef = useRef(null);
  const ttsAudioRef = useRef(null);
  const ttsPlaybackTimerRef = useRef(null);
  const chatSocketRef = useRef(null);
  const chatRecorderRef = useRef(null);
  const chatStreamRef = useRef(null);
  const chatAbortRef = useRef(null);
  const chatTtsAbortRef = useRef(null);
  const chatTtsAudioContextRef = useRef(null);
  const chatTtsDelayTimerRef = useRef(null);
  const chatTtsDelayResolveRef = useRef(null);
  const chatRoomToneRef = useRef(null);
  const chatRoomToneStartIdRef = useRef(0);
  const chatSpeechQueueRef = useRef([]);
  const chatSpeechQueueRunningRef = useRef(false);
  const chatMessagesRef = useRef([]);
  const chatLatestPartialRef = useRef("");
  const chatPartialTimerRef = useRef(null);
  const chatPartialUpdatedAtRef = useRef(0);
  const chatLastSpeechAtRef = useRef(0);
  const chatRespondingRef = useRef(false);
  const chatRequestIdRef = useRef("");
  const chatModelsRequestedRef = useRef(false);
  const chatTurnStartedAtRef = useRef(0);
  const chatTimingRef = useRef(null);
  const chatSpeechActiveRef = useRef(false);
  const lastChatUserTextRef = useRef("");
  const lastChatUserSentAtRef = useRef(0);
  const liveSessionStartedAtMsRef = useRef(0);
  const liveSpeechStartMsRef = useRef(null);
  const liveSpeechEndMsRef = useRef(null);
  const chunkCountRef = useRef(0);
  const byteCountRef = useRef(0);

  const supportedMimeType = useMemo(getSupportedMimeType, []);

  const transcript = useMemo(() => asTranscriptText(transcriptSegments), [transcriptSegments]);
  const visibleTranscript = useMemo(
    () => getNaturalWindow(transcript, visibleWords),
    [transcript, visibleWords],
  );
  const fileTranscript = useMemo(() => {
    if (!fileResult?.result) {
      return "";
    }

    const diarizedSegments = findTranscriptSegmentList(fileResult.result);
    if (diarizedSegments.length) {
      return asDiarizedTranscriptText(diarizedSegments);
    }

    const { text } = readTranscript(JSON.stringify(fileResult.result));
    return text;
  }, [fileResult]);
  const selectedAudioInput = useMemo(
    () => audioInputs.find((input) => input.deviceId === selectedAudioInputId),
    [audioInputs, selectedAudioInputId],
  );
  const systemVoices = useMemo(
    () => ttsVoices.filter((voice) => voice.kind !== "clone"),
    [ttsVoices],
  );
  const clonedVoices = useMemo(
    () => ttsVoices.filter((voice) => voice.kind === "clone"),
    [ttsVoices],
  );
  const chatTimingRows = useMemo(() => buildChatTimingRows(chatTiming), [chatTiming]);
  const slowestChatStage = useMemo(
    () => findSlowestChatStage(chatTimingRows),
    [chatTimingRows],
  );

  const setChatMessagesSynced = (updater) => {
    setChatMessages((current) => {
      const next = typeof updater === "function" ? updater(current) : updater;
      chatMessagesRef.current = next;
      return next;
    });
  };

  const clearChatPartialTimer = () => {
    if (chatPartialTimerRef.current) {
      clearTimeout(chatPartialTimerRef.current);
      chatPartialTimerRef.current = null;
    }
  };

  const addChatLatencyEvent = (label, detail = "") => {
    const now = performance.now();
    const start = chatTurnStartedAtRef.current || now;
    const event = {
      id: crypto.randomUUID(),
      label,
      detail,
      at: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
      elapsedMs: now - start,
    };
    setChatLatencyEvents((current) => [...current, event].slice(-32));
  };

  const resetChatTiming = () => {
    chatTimingRef.current = null;
    chatSpeechActiveRef.current = false;
    setChatTiming(null);
  };

  const startChatTiming = () => {
    const timing = createChatTiming();
    timing.speechStartedAt = performance.now();
    chatTimingRef.current = timing;
    chatTurnStartedAtRef.current = timing.speechStartedAt;
    setChatLatencyEvents([]);
    setChatTiming(timing);
  };

  const markChatTiming = (key, detail = "", options = {}) => {
    const now = performance.now();
    const current = chatTimingRef.current ?? createChatTiming();
    if (current.speechStartedAt == null && chatTurnStartedAtRef.current) {
      current.speechStartedAt = chatTurnStartedAtRef.current;
    }
    if (options.overwrite || current[key] == null) {
      current[key] = now;
    }
    if (key === "transcriptReadyAt" && detail) {
      current.transcriptSource = detail;
    }
    if (key === "ttsRequestAt" && detail) {
      current.ttsPreview = detail;
    }
    chatTimingRef.current = { ...current };
    setChatTiming(chatTimingRef.current);
    return now;
  };

  const stopChatRoomTone = (immediate = false) => {
    chatRoomToneStartIdRef.current += 1;
    const bed = chatRoomToneRef.current;
    chatRoomToneRef.current = null;
    bed?.stop?.(immediate).catch(() => {});
  };

  const startChatRoomTone = async () => {
    if (!isChatRoomToneEnabled || chatRoomToneRef.current) {
      return;
    }

    const startId = chatRoomToneStartIdRef.current + 1;
    chatRoomToneStartIdRef.current = startId;
    try {
      const bed = await createRoomToneBed(chatRoomToneVolume);
      if (chatRoomToneStartIdRef.current !== startId) {
        await bed.stop(true);
        return;
      }
      chatRoomToneRef.current = bed;
    } catch {
      chatRoomToneRef.current = null;
    }
  };

  const stopChatTts = () => {
    if (chatTtsDelayTimerRef.current) {
      clearTimeout(chatTtsDelayTimerRef.current);
      chatTtsDelayTimerRef.current = null;
    }
    chatTtsDelayResolveRef.current?.();
    chatTtsDelayResolveRef.current = null;
    chatSpeechQueueRef.current = [];
    chatTtsAbortRef.current?.abort();
    chatTtsAudioContextRef.current?.close?.().catch(() => {});
    chatTtsAbortRef.current = null;
    chatTtsAudioContextRef.current = null;
    chatSpeechQueueRunningRef.current = false;
    setIsChatSpeaking(false);
  };

  const waitForChatPlayback = (durationMs) =>
    new Promise((resolve) => {
      if (durationMs <= 0) {
        resolve();
        return;
      }
      chatTtsDelayResolveRef.current = resolve;
      chatTtsDelayTimerRef.current = setTimeout(() => {
        chatTtsDelayTimerRef.current = null;
        chatTtsDelayResolveRef.current = null;
        resolve();
      }, durationMs);
    });

  const processChatSpeechQueue = async () => {
    if (chatSpeechQueueRunningRef.current) {
      return;
    }

    chatSpeechQueueRunningRef.current = true;
    setIsChatSpeaking(true);

    try {
      while (chatSpeechQueueRef.current.length) {
        const text = chatSpeechQueueRef.current.shift();
        if (!text?.trim()) {
          continue;
        }

        chatTtsAbortRef.current?.abort();
        const abortController = new AbortController();
        chatTtsAbortRef.current = abortController;
        setChatStatus("Speaking");
        addChatLatencyEvent("TTS request", text.slice(0, 48));
        markChatTiming("ttsRequestAt", text.slice(0, 48));
        let sawFirstTtsChunk = false;

        const response = await fetch(`${proxyHttpUrl}/api/tts/synthesize/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          signal: abortController.signal,
          body: JSON.stringify({
            text,
            voice: ttsVoice,
            language: ttsLanguage,
            stream_format: "pcm_s16le",
          }),
        });

        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.error ?? payload.result?.error ?? response.statusText);
        }

        const streamStats = await playPcmStream({
          response,
          signal: abortController.signal,
          onAudioContext: (audioContext) => {
            chatTtsAudioContextRef.current = audioContext;
          },
          onChunk: ({ chunkCount }) => {
            if (!sawFirstTtsChunk) {
              sawFirstTtsChunk = true;
              addChatLatencyEvent("First audio chunk");
              markChatTiming("firstTtsAudioAt");
            }
            setChatStatus(`Speaking chunk ${chunkCount}`);
          },
        });

        if (abortController.signal.aborted) {
          break;
        }

        await waitForChatPlayback(Math.max(0, streamStats.playbackDelayMs - 80));
        markChatTiming("ttsDoneAt", "", { overwrite: true });
        chatTtsAudioContextRef.current?.close?.().catch(() => {});
        chatTtsAudioContextRef.current = null;
      }
    } catch (caught) {
      if (!(caught instanceof DOMException && caught.name === "AbortError")) {
        setChatError(caught instanceof Error ? caught.message : "TTS playback failed.");
      }
    } finally {
      chatTtsAbortRef.current = null;
      chatSpeechQueueRunningRef.current = false;
      setIsChatSpeaking(false);
      setChatStatus((current) => (current.startsWith("Speaking") ? "Listening" : current));
    }
  };

  const queueChatSpeech = (text) => {
    const cleaned = repairTextEncoding(text).replace(/\s+/g, " ").trim();
    if (!cleaned) {
      return;
    }
    chatSpeechQueueRef.current.push(cleaned);
    void processChatSpeechQueue();
  };

  const renderVoiceOptions = () => (
    <>
      <optgroup label="System voices">
        <option value="alba">alba</option>
        {systemVoices.filter((voice) => voice.id !== "alba").map((voice) => (
          <option key={`system-${voice.id}`} value={voice.id}>
            {voice.label}
          </option>
        ))}
      </optgroup>
      {clonedVoices.length > 0 && (
        <optgroup label="Clones">
          {clonedVoices.map((voice) => (
            <option key={`clone-${voice.id}`} value={voice.id}>
              {voice.label}
            </option>
          ))}
        </optgroup>
      )}
    </>
  );

  const loadTtsVoices = async () => {
    setTtsError("");
    try {
      const response = await fetch(`${proxyHttpUrl}/api/tts/voices`);
      const payload = await response.json();
      if (!response.ok) {
        setTtsError(payload.error ?? payload.result?.error ?? response.statusText);
        return;
      }

      const voices = normalizeVoices(payload);
      const languages = normalizeLanguages(payload);
      setTtsVoices(voices);
      setTtsLanguages(languages);
      if (voices.length && !voices.some((voice) => voice.id === ttsVoice)) {
        setTtsVoice(voices[0].id);
      }
      if (languages.length && !languages.includes(ttsLanguage)) {
        setTtsLanguage(languages[0]);
      }
    } catch (caught) {
      setTtsError(caught instanceof Error ? caught.message : "Unable to load TTS voices.");
    }
  };

  const loadAudioInputs = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices
      .filter((device) => device.kind === "audioinput")
      .map((device, index) => ({
        deviceId: device.deviceId,
        groupId: device.groupId,
        label: getAudioDeviceLabel(device, index),
      }));

    setAudioInputs(inputs);

    if (selectedAudioInputId && !inputs.some((input) => input.deviceId === selectedAudioInputId)) {
      setSelectedAudioInputId("");
    }
  };

  const loadChatModels = async () => {
    setChatModelsError("");
    setIsChatModelsLoading(true);
    chatModelsRequestedRef.current = true;

    try {
      const response = await fetch(`${proxyHttpUrl}/api/chat/models`);
      const payload = await response.json();
      if (!response.ok) {
        setChatModelsError(payload.error ?? payload.result?.error ?? response.statusText);
        return;
      }

      const models = normalizeChatModels(payload);
      setChatModels(models);
      if (models.length && !models.includes(chatModel)) {
        setChatModel(models.includes(DEFAULT_CHAT_MODEL) ? DEFAULT_CHAT_MODEL : models[0]);
      }
      if (!models.length) {
        setChatModelsError("The models endpoint returned no model IDs.");
      }
    } catch (caught) {
      setChatModelsError(caught instanceof Error ? caught.message : "Unable to load chat models.");
    } finally {
      setIsChatModelsLoading(false);
    }
  };

  const buildAudioCaptureConstraints = () => {
    const isMicrophone = audioInputType === "microphone";
    return {
      ...(selectedAudioInputId ? { deviceId: { exact: selectedAudioInputId } } : {}),
      channelCount: 1,
      echoCancellation: isMicrophone,
      noiseSuppression: isMicrophone,
      autoGainControl: isMicrophone,
    };
  };

  const addEvent = (event) => {
    const stampedEvent = {
      time: formatTime(),
      ...event,
    };

    setEvents((current) => {
      const latest = current.at(0);
      if (
        stampedEvent.kind === "transcript" &&
        stampedEvent.final === false &&
        latest?.kind === "transcript" &&
        latest.final === false
      ) {
        return [stampedEvent, ...current.slice(1)];
      }

      return [stampedEvent, ...current];
    });
  };

  const addTranscriptSegment = ({ text, parsed, payload, timing = {} }) => {
    const cleanedText = repairTextEncoding(text).replace(/\s+/g, " ").trim();
    if (!cleanedText) {
      return;
    }

    const final = isFinalTranscript(parsed);
    const speaker = readSpeaker(parsed);
    const speakerSegment = readSpeakerSegment(parsed);
    const startMs = readStartMs(parsed) ?? timing.startMs ?? null;
    const endMs = readEndMs(parsed) ?? timing.endMs ?? null;
    const segment = {
      id: crypto.randomUUID(),
      sequence: transcriptSegments.length + 1,
      receivedAt: new Date().toISOString(),
      text: cleanedText,
      speaker,
      speakerSegment,
      startMs,
      endMs,
      final,
      raw: payload,
      parsed,
    };

    setTranscriptSegments((current) => {
      const last = current.at(-1);
      if (last && !last.final) {
        return [
          ...current.slice(0, -1),
          {
            ...segment,
            sequence: last.sequence,
          },
        ];
      }

      return [
        ...current,
        {
          ...segment,
          sequence: current.length + 1,
        },
      ];
    });
  };

  const exportSessionJson = () => {
    const transcriptSegmentsForExport = transcriptSegments.map((segment, index) => ({
      ...segment,
      sequence: index + 1,
    }));
    const subtitleCues = buildSubtitleCues(transcriptSegmentsForExport, sessionStartedAt);

    const session = {
      exportedAt: new Date().toISOString(),
        source: "live-asr",
        format: "speaker-subtitle-json",
        timestampFormat: "MM:SS or HH:MM:SS",
        inputType: audioInputType,
        expectedSpeakers: Number(diarizationSpeakerCount) || null,
        subtitles: subtitleCues,
      };

    downloadJson(session, buildSessionFileName());
  };

  const transcribeFile = async () => {
    const audioUrl = fileAudioUrl.trim();
    const isUpload = fileInputMode === "upload";
    setFileError("");

    if (!isUpload && !audioUrl) {
      setFileError("Enter an audio or video URL.");
      return;
    }
    if (isUpload && !fileAudioFile) {
      setFileError("Choose a local audio or video file.");
      return;
    }

    setIsFileTranscribing(true);
    setFileStatus(isUpload ? "Uploading" : "Submitting");
    setFileResult(null);

    try {
      const requestOptions = isUpload
        ? (() => {
            const formData = new FormData();
            formData.append("audio", fileAudioFile);
            formData.append("mode", "asr");
            formData.append("enable_diarization", "true");
            for (const [key, value] of Object.entries(
              buildDiarizationOptions(diarizationSpeakerCount),
            )) {
              formData.append(key, value.toString());
            }
            return {
              method: "POST",
              body: formData,
            };
          })()
        : {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              audio_url: audioUrl,
              mode: "asr",
              enable_diarization: true,
              ...buildDiarizationOptions(diarizationSpeakerCount),
            }),
          };
      const response = await fetch(`${proxyHttpUrl}/api/transcribe`, {
        ...requestOptions,
      });
      const payload = await response.json();

      setFileResult({
        ...payload,
        requestedAt: new Date().toISOString(),
        inputMode: fileInputMode,
        ...(isUpload
          ? {
              source_filename: fileAudioFile.name,
              source_type: fileAudioFile.type || "application/octet-stream",
              source_size: fileAudioFile.size,
            }
          : { audio_url: audioUrl }),
        mode: "asr",
        enable_diarization: true,
        diarization_speakers: Number(diarizationSpeakerCount) || null,
      });
      setFileStatus(response.ok ? "Complete" : `Failed (${response.status})`);
      if (!response.ok) {
        setFileError(payload.error ?? payload.result?.error ?? response.statusText);
      }
    } catch (caught) {
      setFileStatus("Failed");
      setFileError(caught instanceof Error ? caught.message : "Transcription request failed.");
    } finally {
      setIsFileTranscribing(false);
    }
  };

  const exportFileTranscription = () => {
    if (!fileResult) {
      return;
    }
    const diarizationSegments = findTranscriptSegmentList(fileResult.result);
    const fallbackSegments = splitTranscriptIntoSubtitleSegments(fileTranscript);
    const subtitleSourceSegments =
      diarizationSegments.length > 1
        ? diarizationSegments
        : diarizationSegments.length === 1 &&
            (diarizationSegments[0].startMs != null || diarizationSegments[0].endMs != null)
          ? diarizationSegments
          : fallbackSegments;
    const subtitleCues = buildSubtitleCues(subtitleSourceSegments);

    downloadJson(
      {
        exportedAt: new Date().toISOString(),
        source: "file-asr",
        audio_url: fileResult.audio_url,
        source_filename: fileResult.source_filename,
        format: "speaker-subtitle-json",
        timestampFormat: "MM:SS or HH:MM:SS",
        subtitles: subtitleCues,
      },
      buildFileTranscriptionFileName(),
    );
  };

  const synthesizeSpeech = async () => {
    const text = repairTextEncoding(ttsText).trim();
    setTtsError("");

    if (!text) {
      setTtsError("Enter text to synthesize.");
      return;
    }

    setIsTtsSynthesizing(true);
    setIsTtsPlaying(false);
    setTtsStatus("Connecting stream");
    setTtsResult(null);
    setTtsAudioSrc("");
    ttsAbortRef.current?.abort();
    ttsAudioContextRef.current?.close?.().catch(() => {});
    if (ttsPlaybackTimerRef.current) {
      clearTimeout(ttsPlaybackTimerRef.current);
      ttsPlaybackTimerRef.current = null;
    }

    const abortController = new AbortController();
    ttsAbortRef.current = abortController;

    try {
      const response = await fetch(`${proxyHttpUrl}/api/tts/synthesize/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        signal: abortController.signal,
        body: JSON.stringify({
          text,
          voice: ttsVoice,
          language: ttsLanguage,
          stream_format: ttsStreamFormat,
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        setTtsResult({
          ...payload,
          requestedAt: new Date().toISOString(),
          text,
          voice: ttsVoice,
          language: ttsLanguage,
          stream_format: ttsStreamFormat,
        });
        setTtsStatus(`Failed (${response.status})`);
        setTtsError(payload.error ?? payload.result?.error ?? response.statusText);
        return;
      }

      const streamFormat =
        response.headers.get("x-tts-stream-format") || ttsStreamFormat;
      const audioFormat = response.headers.get("x-audio-format") || streamFormat;
      const sampleRate = Number(response.headers.get("x-audio-sample-rate") ?? 24000);

      setTtsStatus(`Streaming ${audioFormat} at ${sampleRate.toLocaleString()} Hz`);

      let streamStats;
      if (streamFormat === "pcm_s16le") {
        streamStats = await playPcmStream({
          response,
          signal: abortController.signal,
          onAudioContext: (audioContext) => {
            ttsAudioContextRef.current = audioContext;
          },
          onChunk: ({ chunkCount, totalBytes, sampleRate: rate }) => {
            setTtsStatus(
              `Playing chunk ${chunkCount} (${totalBytes.toLocaleString()} bytes at ${rate.toLocaleString()} Hz)`,
            );
          },
        });
        if (abortController.signal.aborted) {
          return;
        }
      } else {
        const chunks = [];
        let totalBytes = 0;
        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("Streaming response body is not readable.");
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          if (abortController.signal.aborted) {
            await reader.cancel();
            return;
          }
          if (value) {
            chunks.push(value);
            totalBytes += value.length;
            setTtsStatus(`Receiving WAV stream (${totalBytes.toLocaleString()} bytes)`);
          }
        }

        const blob = new Blob(chunks, { type: "audio/wav" });
        setTtsAudioSrc(URL.createObjectURL(blob));
        setIsTtsPlaying(true);
        streamStats = {
          chunkCount: chunks.length,
          totalBytes,
          sampleRate,
          estimatedDurationMs: null,
        };
      }

      const result = {
        ok: true,
        status: response.status,
        statusText: response.statusText,
        stream: {
          format: streamFormat,
          audioFormat,
          sampleRate,
          chunks: streamStats.chunkCount,
          bytes: streamStats.totalBytes,
          estimatedDurationMs: streamStats.estimatedDurationMs,
        },
        requestedAt: new Date().toISOString(),
        text,
        voice: ttsVoice,
        language: ttsLanguage,
        stream_format: ttsStreamFormat,
      };

      setTtsResult(result);
      if (streamFormat === "pcm_s16le" && streamStats.playbackDelayMs > 250) {
        setIsTtsPlaying(true);
        setTtsStatus("Playing buffered audio");
        ttsPlaybackTimerRef.current = setTimeout(() => {
          ttsPlaybackTimerRef.current = null;
          ttsAudioContextRef.current?.close?.().catch(() => {});
          ttsAudioContextRef.current = null;
          setIsTtsPlaying(false);
          setTtsStatus("Ready");
        }, streamStats.playbackDelayMs);
      } else {
        setTtsStatus("Ready");
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        setTtsStatus("Stopped");
        return;
      }
      setTtsStatus("Failed");
      setTtsError(caught instanceof Error ? caught.message : "TTS request failed.");
    } finally {
      setIsTtsSynthesizing(false);
      ttsAbortRef.current = null;
    }
  };

  const stopTts = () => {
    if (ttsPlaybackTimerRef.current) {
      clearTimeout(ttsPlaybackTimerRef.current);
      ttsPlaybackTimerRef.current = null;
    }
    clearChatPartialTimer();
    ttsAbortRef.current?.abort();
    ttsAudioContextRef.current?.close?.().catch(() => {});
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      ttsAudioRef.current.currentTime = 0;
    }
    ttsAbortRef.current = null;
    ttsAudioContextRef.current = null;
    setIsTtsSynthesizing(false);
    setIsTtsPlaying(false);
    setTtsStatus("Stopped");
  };

  const exportTtsResult = () => {
    if (!ttsResult) {
      return;
    }
    downloadJson(ttsResult, buildTtsFileName());
  };

  const downloadTtsAudio = async () => {
    const text = repairTextEncoding(ttsText).trim();
    setTtsError("");

    if (!text) {
      setTtsError("Enter text to synthesize.");
      return;
    }

    setIsTtsDownloading(true);
    setTtsStatus("Preparing download");

    try {
      const response = await fetch(`${proxyHttpUrl}/api/tts/synthesize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          voice: ttsVoice,
          language: ttsLanguage,
          response_format: "wav",
        }),
      });
      const payload = await response.json();

      if (!response.ok) {
        setTtsResult({
          ...payload,
          requestedAt: new Date().toISOString(),
          text,
          voice: ttsVoice,
          language: ttsLanguage,
          response_format: "wav",
        });
        setTtsStatus(`Download failed (${response.status})`);
        setTtsError(payload.error ?? payload.result?.error ?? response.statusText);
        return;
      }

      const audioBase64 = findAudioBase64(payload.result ?? payload);
      if (!audioBase64) {
        setTtsResult({
          ...payload,
          requestedAt: new Date().toISOString(),
          text,
          voice: ttsVoice,
          language: ttsLanguage,
          response_format: "wav",
        });
        setTtsStatus("Download failed");
        setTtsError("The TTS response did not include downloadable audio.");
        return;
      }

      const mimeType = findMimeType(payload.result ?? payload) || "audio/wav";
      const extension = mimeType.includes("mpeg")
        ? "mp3"
        : mimeType.includes("ogg")
          ? "ogg"
          : "wav";
      const blob = base64ToBlob(audioBase64, mimeType);
      downloadBlob(blob, buildTtsAudioFileName(extension));
      setTtsResult({
        ...payload,
        requestedAt: new Date().toISOString(),
        text,
        voice: ttsVoice,
        language: ttsLanguage,
        response_format: "wav",
        downloadedAudio: {
          mimeType,
          bytes: blob.size,
        },
      });
      setTtsStatus("Download ready");
    } catch (caught) {
      setTtsStatus("Download failed");
      setTtsError(caught instanceof Error ? caught.message : "TTS download failed.");
    } finally {
      setIsTtsDownloading(false);
    }
  };

  const queueChatTurnFromPartial = (delayMs = chatTurnMode === "fast" ? 900 : 2200) => {
    clearChatPartialTimer();
    const text = repairTextEncoding(chatLatestPartialRef.current).replace(/\s+/g, " ").trim();
    if (!text) {
      return;
    }

    const scheduledPartial = text;
    const scheduledAt = chatPartialUpdatedAtRef.current;
    chatPartialTimerRef.current = setTimeout(() => {
      chatPartialTimerRef.current = null;
      const settledText = repairTextEncoding(chatLatestPartialRef.current)
        .replace(/\s+/g, " ")
        .trim();
      if (
        !settledText ||
        settledText !== scheduledPartial ||
        chatPartialUpdatedAtRef.current !== scheduledAt ||
        chatRespondingRef.current
      ) {
        return;
      }
      chatLatestPartialRef.current = "";
      setChatPartialTranscript("");
      setChatVadStatus("Partial settled");
      addChatLatencyEvent("Partial stable", settledText.slice(0, 48));
      markChatTiming("transcriptReadyAt", "partial stable");
      void sendChatTurn(settledText, { source: "partial fallback" });
    }, delayMs);
  };

  const sendChatTurn = async (spokenText, options = {}) => {
    const text = repairTextEncoding(spokenText).replace(/\s+/g, " ").trim();
    if (!text) {
      return;
    }

    const now = Date.now();
    const isRecentFollowup = now - lastChatUserSentAtRef.current < DUPLICATE_CHAT_TURN_WINDOW_MS;
    if (
      isRecentFollowup &&
      isDuplicateOrOlderUtterance(text, lastChatUserTextRef.current)
    ) {
      clearChatPartialTimer();
      chatLatestPartialRef.current = "";
      chatPartialUpdatedAtRef.current = 0;
      setChatPartialTranscript("");
      setChatVadStatus("Duplicate transcript ignored");
      addChatLatencyEvent("Duplicate ignored", text.slice(0, 48));
      return;
    }

    const shouldReplacePreviousTurn =
      isExpandedUtterance(text, lastChatUserTextRef.current) &&
      isRecentFollowup;

    lastChatUserTextRef.current = text;
    lastChatUserSentAtRef.current = now;
    clearChatPartialTimer();
    chatAbortRef.current?.abort();
    stopChatTts();

    const existingMessages = shouldReplacePreviousTurn
      ? chatMessagesRef.current.slice(0, -2)
      : chatMessagesRef.current;
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      createdAt: new Date().toISOString(),
    };
    const assistantMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "",
      createdAt: new Date().toISOString(),
    };
    const priorMessages = existingMessages;
    setChatMessagesSynced([...priorMessages, userMessage, assistantMessage]);
    chatRespondingRef.current = true;
    setIsChatResponding(true);
    setChatStatus("Thinking");
    setChatError("");
    addChatLatencyEvent("Chat request", options.source ?? "transcript");
    markChatTiming("transcriptReadyAt", options.source ?? "transcript");
    markChatTiming("chatRequestAt", options.source ?? "transcript");

    const abortController = new AbortController();
    const requestId = crypto.randomUUID();
    chatRequestIdRef.current = requestId;
    chatAbortRef.current = abortController;
    let assistantText = "";
    let speechBuffer = "";
    let sawFirstToken = false;
    let sawFirstSpeechChunk = false;

    try {
      const historyMessages = buildAlternatingChatMessages([...priorMessages, userMessage]);
      const apiMessages = [
        ...(chatSystemPrompt.trim()
          ? [{ role: "system", content: chatSystemPrompt.trim() }]
          : []),
        ...historyMessages,
      ];
      addChatLatencyEvent(
        "History packed",
        apiMessages
          .filter((message) => message.role !== "system")
          .map((message) => message.role[0])
          .join("/"),
      );

      const response = await fetch(`${proxyHttpUrl}/api/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        signal: abortController.signal,
        body: JSON.stringify({
          model: chatModel.trim() || DEFAULT_CHAT_MODEL,
          messages: apiMessages,
          stream: true,
          temperature: 0.7,
          ...(Number(chatMaxTokens) > 0
            ? { max_tokens: Math.floor(Number(chatMaxTokens)) }
            : {}),
          ...(Number(chatContextWindow) > 0
            ? { context_window: Math.floor(Number(chatContextWindow)) }
            : {}),
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error ?? payload.result?.error ?? response.statusText);
      }

      await readChatCompletionStream({
        response,
        signal: abortController.signal,
        onDelta: (delta) => {
          if (!sawFirstToken) {
            sawFirstToken = true;
            addChatLatencyEvent("First LLM token");
            markChatTiming("firstLlmTokenAt");
          }
          const cleanedDelta = repairTextEncoding(delta);
          assistantText += cleanedDelta;
          speechBuffer += cleanedDelta;
          setChatMessagesSynced((current) =>
            current.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, text: assistantText }
                : message,
            ),
          );

          const speakable = takeSpeakableText(speechBuffer, false, chatTurnMode);
          if (speakable.text) {
            speechBuffer = speakable.remaining;
            if (!sawFirstSpeechChunk) {
              sawFirstSpeechChunk = true;
              addChatLatencyEvent("First speakable text", speakable.text.slice(0, 48));
              markChatTiming("firstSpeakableAt");
            }
            queueChatSpeech(speakable.text);
          }
          setChatStatus("Replying");
        },
      });

      if (abortController.signal.aborted) {
        return;
      }

      const finalSpeech = takeSpeakableText(speechBuffer, true, chatTurnMode);
      markChatTiming("llmDoneAt", "", { overwrite: true });
      if (finalSpeech.text) {
        queueChatSpeech(finalSpeech.text);
      }
      setChatMessagesSynced((current) =>
        current.map((message) =>
          message.id === assistantMessage.id
            ? { ...message, text: assistantText.trim(), finishedAt: new Date().toISOString() }
            : message,
        ),
      );
      setChatStatus(chatSpeechQueueRef.current.length ? "Speaking" : "Listening");
    } catch (caught) {
      if (chatRequestIdRef.current !== requestId) {
        return;
      }
      if (caught instanceof DOMException && caught.name === "AbortError") {
        setChatStatus("Stopped");
        return;
      }
      setChatError(caught instanceof Error ? caught.message : "Chat request failed.");
      setChatStatus("Chat failed");
    } finally {
      if (chatRequestIdRef.current === requestId) {
        chatRespondingRef.current = false;
        setIsChatResponding(false);
        chatAbortRef.current = null;
      }
    }
  };

  const stopChatAudioCapture = () => {
    try {
      chatRecorderRef.current?.stop?.();
    } catch {
      // MediaRecorder can throw if it is already inactive.
    }
    chatRecorderRef.current = null;

    for (const track of chatStreamRef.current?.getTracks?.() ?? []) {
      track.stop();
    }
    chatStreamRef.current = null;
  };

  const stopVoiceChat = () => {
    clearChatPartialTimer();
    stopChatAudioCapture();
    chatSocketRef.current?.close?.();
    chatSocketRef.current = null;
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    chatRequestIdRef.current = "";
    stopChatTts();
    stopChatRoomTone();
    setIsVoiceChatActive(false);
    chatRespondingRef.current = false;
    setIsChatResponding(false);
    setChatPartialTranscript("");
    chatLatestPartialRef.current = "";
    chatPartialUpdatedAtRef.current = 0;
    chatLastSpeechAtRef.current = 0;
    chatTurnStartedAtRef.current = 0;
    resetChatTiming();
    setChatVadStatus("Stopped");
    setChatStatus("Stopped");
  };

  const startVoiceChat = async () => {
    setChatError("");
    setChatPartialTranscript("");
    chatLatestPartialRef.current = "";
    chatPartialUpdatedAtRef.current = 0;
    chatLastSpeechAtRef.current = 0;
    chatTurnStartedAtRef.current = 0;
    chatRespondingRef.current = false;
    chatRequestIdRef.current = "";
    lastChatUserTextRef.current = "";
    lastChatUserSentAtRef.current = 0;
    resetChatTiming();

    if (!navigator.mediaDevices?.getUserMedia) {
      setChatError("This browser does not support microphone capture.");
      return;
    }

    try {
      setChatStatus("Opening microphone");
      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: buildAudioCaptureConstraints(),
      });
      chatStreamRef.current = audioStream;
      void startChatRoomTone();
      loadAudioInputs().catch(() => {});

      setChatStatus("Connecting VAD websocket");
      const socket = new WebSocket(buildVadWebsocketUrl(proxyUrl, chatTurnMode));
      socket.binaryType = "arraybuffer";
      chatSocketRef.current = socket;

      socket.onopen = () => {
        chatRecorderRef.current = startPcmStream({
          audioStream,
          socket,
          sampleRate: DEFAULT_SAMPLE_RATE,
          onChunk: () => {},
        });
        setIsVoiceChatActive(true);
        setChatStatus("Listening");
        setChatVadStatus("Listening");
      };

      socket.onmessage = async (message) => {
        const payload =
          typeof message.data === "string" ? message.data : await message.data.text?.();
        const { parsed, text, isExplicitTranscript } = readTranscript(payload);
        const type = parsed?.type ?? parsed?.event ?? parsed?.message_type ?? parsed?.status ?? "";
        const normalizedType = type.toString().toLowerCase();

        if (
          normalizedType === "proxy-status" &&
          parsed?.status === "connected" &&
          socket.readyState === WebSocket.OPEN
        ) {
          socket.send(JSON.stringify(buildChatAsrVadConfig(chatTurnMode)));
          setChatVadStatus("Configured ASR VAD");
        }

        if (normalizedType.includes("vad")) {
          setChatVadStatus(type.toString() || "VAD event");
        }
        if (parsed?.is_speech === true) {
          chatLastSpeechAtRef.current = Date.now();
        }
        if (normalizedType.includes("speech_started")) {
          if (chatSpeechActiveRef.current) {
            return;
          }
          chatSpeechActiveRef.current = true;
          startChatTiming();
          addChatLatencyEvent("Speech started");
          if (isChatSpeaking || isChatResponding) {
            chatAbortRef.current?.abort();
            stopChatTts();
            chatRespondingRef.current = false;
            setIsChatResponding(false);
            setChatStatus("Listening");
            addChatLatencyEvent("Barge-in");
          }
        }

        const speechEnded =
          normalizedType.includes("speech_stopped") ||
          normalizedType.includes("speech_ended") ||
          normalizedType.includes("utterance_end") ||
          normalizedType.includes("end_of_speech");
        if (speechEnded) {
          chatSpeechActiveRef.current = false;
        }
        if (speechEnded && chatLatestPartialRef.current.trim()) {
          setChatVadStatus("Speech ended");
          addChatLatencyEvent("Speech ended", chatLatestPartialRef.current.slice(0, 48));
          markChatTiming("speechEndedAt");
          if (chatTurnMode === "fast") {
            clearChatPartialTimer();
            markChatTiming("transcriptReadyAt", "speech ended partial");
            void sendChatTurn(chatLatestPartialRef.current, { source: "speech ended partial" });
          } else {
            queueChatTurnFromPartial(500);
          }
        }

        const isSystemMessage =
          normalizedType.startsWith("proxy-") ||
          type === "proxy-status" ||
          type === "proxy-error" ||
          parsed?.status === "connected" ||
          normalizedType.includes("error");

        if (isSystemMessage) {
          return;
        }

        const cleanedText = repairTextEncoding(text).replace(/\s+/g, " ").trim();
        if (!cleanedText || (!isExplicitTranscript && !normalizedType.includes("transcript"))) {
          return;
        }

        const isCommittedTranscript = normalizedType.includes("committed_transcript");

        if (isCommittedTranscript || isFinalTranscript(parsed)) {
          clearChatPartialTimer();
          chatLatestPartialRef.current = "";
          chatPartialUpdatedAtRef.current = 0;
          setChatPartialTranscript("");
          setChatVadStatus(isCommittedTranscript ? "Committed transcript" : "Utterance ready");
          addChatLatencyEvent(
            isCommittedTranscript ? "Committed transcript" : "Final transcript",
            cleanedText.slice(0, 48),
          );
          markChatTiming(
            "transcriptReadyAt",
            isCommittedTranscript ? "committed transcript" : "final transcript",
          );
          void sendChatTurn(cleanedText, {
            source: isCommittedTranscript ? "committed transcript" : "final transcript",
          });
        } else {
          chatLatestPartialRef.current = cleanedText;
          chatPartialUpdatedAtRef.current = Date.now();
          setChatPartialTranscript(cleanedText);
          setChatVadStatus("Speech detected");
          queueChatTurnFromPartial();
        }
      };

      socket.onerror = () => {
        setChatError("Voice chat websocket error. Check the proxy terminal and .env auth value.");
      };

      socket.onclose = (event) => {
        const reason = event.reason ? `: ${event.reason}` : "";
        setChatStatus(`Closed (${event.code})${reason}`);
        setIsVoiceChatActive(false);
        stopChatAudioCapture();
        chatSocketRef.current = null;
      };
    } catch (caught) {
      stopVoiceChat();
      setChatStatus("Idle");
      setChatError(caught instanceof Error ? caught.message : "Unable to start voice chat.");
    }
  };

  const clearVoiceChat = () => {
    clearChatPartialTimer();
    setChatMessagesSynced([]);
    setChatLatencyEvents([]);
    setChatPartialTranscript("");
    setChatError("");
    chatLatestPartialRef.current = "";
    chatPartialUpdatedAtRef.current = 0;
    chatLastSpeechAtRef.current = 0;
    chatTurnStartedAtRef.current = 0;
    resetChatTiming();
    chatRequestIdRef.current = "";
    lastChatUserTextRef.current = "";
    lastChatUserSentAtRef.current = 0;
  };

  const exportVoiceChatJson = () => {
    downloadJson(
      {
        exportedAt: new Date().toISOString(),
        model: chatModel,
        maxTokens: Number(chatMaxTokens) > 0 ? Math.floor(Number(chatMaxTokens)) : null,
        contextWindow:
          Number(chatContextWindow) > 0 ? Math.floor(Number(chatContextWindow)) : null,
        systemPrompt: chatSystemPrompt,
        messages: chatMessages,
      },
      buildVoiceChatFileName(),
    );
  };

  const createVoiceClone = async () => {
    const id = cloneId.trim();
    setCloneError("");

    if (!id) {
      setCloneError("Enter a clone id.");
      return;
    }
    if (!cloneFile) {
      setCloneError("Choose a WAV, MP3, or OGG reference file.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", cloneFile);
    formData.append("clone_id", id);
    formData.append("language", cloneLanguage.trim() || "english");

    setIsCloningVoice(true);
    setCloneStatus("Uploading reference");
    setCloneResult(null);

    try {
      const response = await fetch(`${proxyHttpUrl}/api/tts/voices/clone`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      const result = {
        ...payload,
        requestedAt: new Date().toISOString(),
        clone_id: id,
        language: cloneLanguage.trim() || "english",
        source_filename: cloneFile.name,
      };
      setCloneResult(result);
      setCloneStatus(response.ok ? "Clone saved" : `Clone failed (${response.status})`);
      if (!response.ok) {
        setCloneError(payload.error ?? payload.result?.error ?? response.statusText);
        return;
      }

      setTtsVoice(id);
      setDeleteCloneId(id);
      await loadTtsVoices();
    } catch (caught) {
      setCloneStatus("Clone failed");
      setCloneError(caught instanceof Error ? caught.message : "Voice clone request failed.");
    } finally {
      setIsCloningVoice(false);
    }
  };

  const deleteVoiceClone = async () => {
    const id = deleteCloneId.trim();
    setCloneError("");

    if (!id) {
      setCloneError("Enter a clone id to delete.");
      return;
    }

    setIsDeletingClone(true);
    setCloneStatus("Deleting clone");
    setCloneResult(null);

    try {
      const response = await fetch(
        `${proxyHttpUrl}/api/tts/voices/clone/${encodeURIComponent(id)}`,
        {
          method: "DELETE",
        },
      );
      const payload = await response.json();
      setCloneResult({
        ...payload,
        requestedAt: new Date().toISOString(),
        clone_id: id,
      });
      setCloneStatus(response.ok ? "Clone deleted" : `Delete failed (${response.status})`);
      if (!response.ok) {
        setCloneError(payload.error ?? payload.result?.error ?? response.statusText);
        return;
      }

      if (ttsVoice === id) {
        setTtsVoice("alba");
      }
      await loadTtsVoices();
    } catch (caught) {
      setCloneStatus("Delete failed");
      setCloneError(caught instanceof Error ? caught.message : "Voice delete request failed.");
    } finally {
      setIsDeletingClone(false);
    }
  };

  const clearSession = () => {
    setEvents([]);
    setTranscriptSegments([]);
    setIncomingCount(0);
    setLastIncoming("");
    setSessionStartedAt(null);
    setSessionEndedAt(null);
    liveSessionStartedAtMsRef.current = 0;
    liveSpeechStartMsRef.current = null;
    liveSpeechEndMsRef.current = null;
  };

  const stopAudioCapture = () => {
    try {
      recorderRef.current?.stop?.();
    } catch {
      // MediaRecorder can throw if it is already inactive.
    }
    recorderRef.current = null;

    for (const track of streamRef.current?.getTracks?.() ?? []) {
      track.stop();
    }
    streamRef.current = null;
  };

  const closeSocket = () => {
    socketRef.current?.close?.();
    socketRef.current = null;
  };

  const cleanup = () => {
    if (finalizeTimerRef.current) {
      clearTimeout(finalizeTimerRef.current);
      finalizeTimerRef.current = null;
    }
    if (ttsPlaybackTimerRef.current) {
      clearTimeout(ttsPlaybackTimerRef.current);
      ttsPlaybackTimerRef.current = null;
    }
    ttsAbortRef.current?.abort();
    ttsAudioContextRef.current?.close?.().catch(() => {});
    chatAbortRef.current?.abort();
    chatSocketRef.current?.close?.();
    stopChatAudioCapture();
    stopChatTts();
    stopChatRoomTone(true);
    stopAudioCapture();
    closeSocket();
  };

  useEffect(() => cleanup, []);

  useEffect(() => {
    chatMessagesRef.current = chatMessages;
  }, [chatMessages]);

  useEffect(() => {
    loadAudioInputs().catch(() => {});

    navigator.mediaDevices?.addEventListener?.("devicechange", loadAudioInputs);
    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", loadAudioInputs);
    };
  }, [selectedAudioInputId]);

  useEffect(() => {
    if ((activeTab === "tts" || activeTab === "chat" || activeTab === "clone") && !ttsVoices.length) {
      loadTtsVoices();
    }
  }, [activeTab, ttsVoices.length]);

  useEffect(() => {
    if (
      activeTab === "chat" &&
      !chatModels.length &&
      !isChatModelsLoading &&
      !chatModelsRequestedRef.current
    ) {
      loadChatModels();
    }
  }, [activeTab, chatModels.length, isChatModelsLoading]);

  useEffect(() => {
    chatRoomToneRef.current?.setVolume?.(chatRoomToneVolume);
  }, [chatRoomToneVolume]);

  useEffect(() => {
    if (isVoiceChatActive && isChatRoomToneEnabled) {
      void startChatRoomTone();
      return;
    }
    stopChatRoomTone();
  }, [isVoiceChatActive, isChatRoomToneEnabled]);

  const startStreaming = async () => {
    setError("");
    if (finalizeTimerRef.current) {
      clearTimeout(finalizeTimerRef.current);
      finalizeTimerRef.current = null;
    }
    setIsFinishing(false);

    if (!sessionStartedAt || sessionEndedAt) {
      const startedAt = new Date();
      setSessionStartedAt(startedAt.toISOString());
      setSessionEndedAt(null);
      setTranscriptSegments([]);
      setEvents([]);
      setIncomingCount(0);
      setLastIncoming("");
      liveSessionStartedAtMsRef.current = startedAt.getTime();
    }
    liveSpeechStartMsRef.current = null;
    liveSpeechEndMsRef.current = null;

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError("This browser does not support microphone capture with MediaRecorder.");
      return;
    }

    try {
      setStatus("Opening microphone");
      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: buildAudioCaptureConstraints(),
      });
      streamRef.current = audioStream;
      loadAudioInputs().catch(() => {});

      setStatus("Connecting websocket");
      const websocketUrl = buildDiarizedAsrWebsocketUrl(proxyUrl, diarizationSpeakerCount);
      const socket = new WebSocket(websocketUrl);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => {
        chunkCountRef.current = 0;
        byteCountRef.current = 0;

        if (audioMode === "pcm") {
          recorderRef.current = startPcmStream({
            audioStream,
            socket,
            sampleRate,
            onChunk: (byteLength) => {
              chunkCountRef.current += 1;
              byteCountRef.current += byteLength;
              if (chunkCountRef.current === 1 || chunkCountRef.current % 20 === 0) {
                addEvent({
                  kind: "system",
                  text: `Sent ${chunkCountRef.current} PCM chunks (${byteCountRef.current.toLocaleString()} bytes).`,
                });
              }
            },
          });
        } else {
          const recorder = new MediaRecorder(
            audioStream,
            supportedMimeType ? { mimeType: supportedMimeType } : undefined,
          );

          recorder.ondataavailable = async (event) => {
            if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
              chunkCountRef.current += 1;
              byteCountRef.current += event.data.size;
              if (chunkCountRef.current === 1 || chunkCountRef.current % 20 === 0) {
                addEvent({
                  kind: "system",
                  text: `Sent ${chunkCountRef.current} WebM chunks (${byteCountRef.current.toLocaleString()} bytes).`,
                });
              }
              socket.send(await event.data.arrayBuffer());
            }
          };

          recorder.start(250);
          recorderRef.current = recorder;
        }

        setIsStreaming(true);
        setStatus("Streaming audio");
        addEvent({
          kind: "system",
          text:
            audioMode === "pcm"
              ? `Streaming mono PCM s16le at ${sampleRate.toLocaleString()} Hz with diarization.`
              : `Streaming ${supportedMimeType || "browser default audio"} chunks with diarization.`,
        });
      };

      socket.onmessage = async (message) => {
        const payload =
          typeof message.data === "string" ? message.data : await message.data.text?.();
        setIncomingCount((current) => current + 1);
        setLastIncoming(payload || "<empty websocket message>");
        console.log("ASR websocket message", payload);

        const { parsed, text, isExplicitTranscript } = readTranscript(payload);
        const type = parsed?.type ?? parsed?.event ?? parsed?.message_type ?? parsed?.status ?? "";
        const normalizedType = type.toString().toLowerCase();
        const speaker = readSpeaker(parsed);
        const nowOffsetMs = Math.max(
          0,
          Date.now() - (liveSessionStartedAtMsRef.current || Date.now()),
        );
        if (normalizedType.includes("speech_started")) {
          liveSpeechStartMsRef.current = nowOffsetMs;
          liveSpeechEndMsRef.current = null;
        }
        const speechEnded =
          normalizedType.includes("speech_stopped") ||
          normalizedType.includes("speech_ended") ||
          normalizedType.includes("utterance_end") ||
          normalizedType.includes("end_of_speech");
        if (speechEnded) {
          liveSpeechEndMsRef.current = nowOffsetMs;
        }
        const isSystemMessage =
          normalizedType.startsWith("proxy-") ||
          type === "proxy-status" ||
          type === "proxy-error" ||
          parsed?.status === "connected" ||
          normalizedType.startsWith("session_") ||
          normalizedType.includes("error");

        const finalTranscript = isFinalTranscript(parsed);
        addEvent({
          kind: isSystemMessage ? "system" : "transcript",
          text,
          speaker: isSystemMessage ? "" : speaker,
          final: isSystemMessage ? undefined : finalTranscript,
        });

        if (!isSystemMessage && (isExplicitTranscript || normalizedType.includes("transcript"))) {
          addTranscriptSegment({
            text,
            parsed,
            payload,
            timing: {
              startMs: liveSpeechStartMsRef.current,
              endMs: liveSpeechEndMsRef.current,
            },
          });
          if (finalTranscript) {
            liveSpeechStartMsRef.current = null;
            liveSpeechEndMsRef.current = null;
          }
        }
      };

      socket.onerror = () => {
        setError("Websocket error. Check the proxy terminal and .env auth value.");
      };

      socket.onclose = (event) => {
        const reason = event.reason ? `: ${event.reason}` : "";
        setStatus(`Closed (${event.code})${reason}`);
        setSessionEndedAt((current) => current ?? new Date().toISOString());
        setIsStreaming(false);
        setIsFinishing(false);
        stopAudioCapture();
        socketRef.current = null;
      };
    } catch (caught) {
      cleanup();
      setIsStreaming(false);
      setStatus("Idle");
      setError(caught instanceof Error ? caught.message : "Unable to start streaming.");
    }
  };

  const stopStreaming = () => {
    stopAudioCapture();
    setIsStreaming(false);
    setIsFinishing(true);
    setStatus("Finishing transcript");
    setSessionEndedAt((current) => current ?? new Date().toISOString());
    addEvent({
      kind: "system",
      text: `Stopped microphone. Waiting ${Math.round(FINALIZE_GRACE_MS / 1000)}s for final transcript messages.`,
    });

    finalizeTimerRef.current = setTimeout(() => {
      finalizeTimerRef.current = null;
      closeSocket();
      setIsFinishing(false);
      setStatus("Stopped");
    }, FINALIZE_GRACE_MS);
  };

  const headerStatus =
    activeTab === "chat"
      ? chatStatus
      : activeTab === "clone"
        ? cloneStatus
        : activeTab === "tts"
          ? ttsStatus
          : status;
  const headerActive =
    activeTab === "chat"
      ? isVoiceChatActive || isChatResponding || isChatSpeaking
      : activeTab === "clone"
        ? isCloningVoice || isDeletingClone
        : activeTab === "tts"
          ? isTtsSynthesizing || isTtsPlaying
          : isStreaming || isFinishing;
  const headerTitle =
    activeTab === "chat"
      ? "Voice Chat Stream"
      : activeTab === "clone"
        ? "Voice Clone"
        : activeTab === "tts"
          ? "Text To Speech"
          : activeTab === "file"
            ? "File Transcription"
            : "Websocket Audio Stream";

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Live ASR</p>
          <h1>{headerTitle}</h1>
        </div>
        <div className={`status ${headerActive ? "active" : ""}`}>
          <span />
          {headerStatus}
        </div>
      </section>

      <section className="tabs" aria-label="Transcription mode">
        <button
          className={activeTab === "live" ? "selected" : ""}
          onClick={() => setActiveTab("live")}
          type="button"
        >
          Live
        </button>
        <button
          className={activeTab === "file" ? "selected" : ""}
          onClick={() => setActiveTab("file")}
          type="button"
        >
          File
        </button>
        <button
          className={activeTab === "tts" ? "selected" : ""}
          onClick={() => setActiveTab("tts")}
          type="button"
        >
          TTS
        </button>
        <button
          className={activeTab === "clone" ? "selected" : ""}
          onClick={() => setActiveTab("clone")}
          type="button"
        >
          Clone
        </button>
        <button
          className={activeTab === "chat" ? "selected" : ""}
          onClick={() => setActiveTab("chat")}
          type="button"
        >
          Chat
        </button>
      </section>

      {activeTab === "live" ? (
        <>
          <section className="panel controls">
        <label>
          <span>Local websocket proxy</span>
          <input
            value={proxyUrl}
            onChange={(event) => setProxyUrl(event.target.value)}
            placeholder={LOCAL_PROXY_URL}
          />
        </label>

        <label>
          <span>Audio input</span>
          <select
            value={selectedAudioInputId}
            onChange={(event) => setSelectedAudioInputId(event.target.value)}
            disabled={isStreaming || isFinishing}
          >
            <option value="">Default microphone</option>
            {audioInputs.map((input) => (
              <option key={input.deviceId} value={input.deviceId}>
                {input.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Input type</span>
          <select
            value={audioInputType}
            onChange={(event) => {
              setAudioInputType(event.target.value);
              if (event.target.value === "system") {
                setAudioMode("pcm");
              }
            }}
            disabled={isStreaming || isFinishing}
          >
            <option value="microphone">Microphone speech</option>
            <option value="system">Computer / loopback audio</option>
          </select>
        </label>

        <label>
          <span>Expected speakers</span>
          <select
            value={diarizationSpeakerCount}
            onChange={(event) => setDiarizationSpeakerCount(event.target.value)}
            disabled={isStreaming || isFinishing}
          >
            <option value="">Auto</option>
            <option value="1">1 speaker</option>
            <option value="2">2 speakers</option>
            <option value="3">3 speakers</option>
            <option value="4">4 speakers</option>
          </select>
        </label>

        <label>
          <span>Audio format</span>
          <select
            value={audioMode}
            onChange={(event) => setAudioMode(event.target.value)}
            disabled={isStreaming || audioInputType === "system"}
          >
            <option value="pcm">PCM s16le mono</option>
            {audioInputType !== "system" && <option value="webm">WebM Opus</option>}
          </select>
        </label>

        {audioMode === "pcm" && (
          <label>
            <span>Sample rate</span>
            <input
              min="8000"
              max="48000"
              step="1000"
              type="number"
              value={sampleRate}
              onChange={(event) => setSampleRate(Number(event.target.value))}
              disabled={isStreaming}
            />
          </label>
        )}

        <label>
          <span>Display words</span>
          <input
            min="10"
            max="200"
            step="5"
            type="number"
            value={visibleWords}
            onChange={(event) => setVisibleWords(Number(event.target.value))}
          />
        </label>

        <div className="buttonRow">
          <button onClick={startStreaming} disabled={isStreaming || isFinishing}>
            Start streaming
          </button>
          <button className="secondary" onClick={stopStreaming} disabled={!isStreaming || isFinishing}>
            Stop audio
          </button>
        </div>

        {error && <p className="error">{error}</p>}
          </section>

          <section className="workspace">
            <div className="panel transcript">
              <div className="sectionHeader">
                <div>
                  <h2>Transcript</h2>
                  <span>
                    {transcriptSegments.length} segments · {incomingCount} incoming
                  </span>
                </div>
                <div className="headerActions">
                  <button
                    className="textButton"
                    onClick={exportSessionJson}
                    disabled={!transcriptSegments.length && !events.length}
                  >
                    Export JSON
                  </button>
                  <button className="textButton" onClick={clearSession}>
                    Clear
                  </button>
                </div>
              </div>
              <p className={visibleTranscript ? "" : "empty"}>
                {visibleTranscript || "Start streaming to see recognized text here."}
              </p>
              {!visibleTranscript && incomingCount > 0 && (
                <pre className="rawPreview">{lastIncoming}</pre>
              )}
              {!visibleTranscript && !incomingCount && events.length > 0 && (
                <p className="hint">
                  Audio chunks are being sent, but no ASR transcript messages have arrived yet.
                </p>
              )}
            </div>

            <div className="panel events">
              <div className="sectionHeader">
                <h2>Messages</h2>
                <span>{events.length}</span>
              </div>
              <div className="eventList">
                {events.map((event, index) => (
                  <article key={`${event.time}-${index}`} className={event.kind}>
                    <time>{event.time}</time>
                    <p>
                      {event.speaker ? `${formatSpeaker(event.speaker)}: ` : ""}
                      {event.text}
                    </p>
                  </article>
                ))}
                {!events.length && <p className="empty">No websocket messages yet.</p>}
              </div>
            </div>
          </section>
        </>
      ) : activeTab === "file" ? (
        <section className="fileMode">
          <div className="panel fileControls">
            <label>
              <span>Source</span>
              <select
                value={fileInputMode}
                onChange={(event) => setFileInputMode(event.target.value)}
                disabled={isFileTranscribing}
              >
                <option value="url">Remote URL</option>
                <option value="upload">Local file upload</option>
              </select>
            </label>
            {fileInputMode === "url" ? (
              <label>
                <span>Audio or video URL</span>
                <input
                  value={fileAudioUrl}
                  onChange={(event) => setFileAudioUrl(event.target.value)}
                  placeholder="https://example.com/audio-or-video.mp4"
                />
              </label>
            ) : (
              <label>
                <span>Audio or video file</span>
                <input
                  type="file"
                  accept="audio/*,video/*,.wav,.mp3,.ogg,.m4a,.mp4,.webm,.mov"
                  onChange={(event) => setFileAudioFile(event.target.files?.[0] ?? null)}
                  disabled={isFileTranscribing}
                />
              </label>
            )}
            <label>
              <span>Expected speakers</span>
              <select
                value={diarizationSpeakerCount}
                onChange={(event) => setDiarizationSpeakerCount(event.target.value)}
                disabled={isFileTranscribing}
              >
                <option value="">Auto</option>
                <option value="1">1 speaker</option>
                <option value="2">2 speakers</option>
                <option value="3">3 speakers</option>
                <option value="4">4 speakers</option>
              </select>
            </label>
            <div className="buttonRow">
              <button onClick={transcribeFile} disabled={isFileTranscribing}>
                {isFileTranscribing ? "Transcribing" : "Transcribe"}
              </button>
              <button
                className="secondary"
                onClick={exportFileTranscription}
                disabled={!fileResult}
              >
                Export JSON
              </button>
            </div>
            {fileError && <p className="error">{fileError}</p>}
          </div>

          <div className="panel transcript fileResult">
            <div className="sectionHeader">
              <div>
                <h2>File Transcript</h2>
                <span>{fileStatus}</span>
              </div>
            </div>
            <p className={fileTranscript ? "" : "empty"}>
              {fileTranscript || "Submit an audio or video URL, or upload a local file to transcribe it."}
            </p>
            {fileResult && <pre className="rawPreview">{JSON.stringify(fileResult, null, 2)}</pre>}
          </div>
        </section>
      ) : activeTab === "tts" ? (
        <section className="fileMode">
          <div className="panel ttsControls">
            <label className="textAreaLabel">
              <span>Text</span>
              <textarea
                value={ttsText}
                onChange={(event) => setTtsText(event.target.value)}
                placeholder="Type something to read aloud."
                rows={6}
              />
            </label>

            <div className="ttsOptions">
              <label>
                <span>Voice</span>
                <select
                  value={ttsVoice}
                  onChange={(event) => setTtsVoice(event.target.value)}
                >
                  {renderVoiceOptions()}
                </select>
              </label>

              <label>
                <span>Language</span>
                <select
                  value={ttsLanguage}
                  onChange={(event) => setTtsLanguage(event.target.value)}
                >
                  <option value="english">english</option>
                  {ttsLanguages.filter((language) => language !== "english").map((language) => (
                    <option key={language} value={language}>
                      {language}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>Stream format</span>
                <select
                  value={ttsStreamFormat}
                  onChange={(event) => setTtsStreamFormat(event.target.value)}
                >
                  <option value="pcm_s16le">pcm_s16le</option>
                  <option value="wav">wav</option>
                </select>
              </label>
            </div>

            <div className="buttonRow">
              <button onClick={synthesizeSpeech} disabled={isTtsSynthesizing || isTtsPlaying}>
                Read
              </button>
              <button
                className="secondary"
                onClick={downloadTtsAudio}
                disabled={isTtsDownloading}
              >
                {isTtsDownloading ? "Downloading" : "Download WAV"}
              </button>
              <button
                className="secondary"
                onClick={stopTts}
                disabled={!isTtsSynthesizing && !isTtsPlaying && !ttsAudioSrc}
              >
                Stop
              </button>
              <button className="secondary" onClick={loadTtsVoices}>
                Get voices
              </button>
              <button className="secondary" onClick={exportTtsResult} disabled={!ttsResult}>
                Export JSON
              </button>
            </div>
            {ttsError && <p className="error">{ttsError}</p>}
          </div>

          <div className="panel transcript fileResult">
            <div className="sectionHeader">
              <div>
                <h2>Speech</h2>
                <span>{ttsStatus}</span>
              </div>
            </div>
            {ttsAudioSrc ? (
              <audio
                ref={ttsAudioRef}
                className="audioPlayer"
                controls
                autoPlay
                src={ttsAudioSrc}
                onEnded={() => {
                  setIsTtsPlaying(false);
                  setTtsStatus("Ready");
                }}
                onPause={() => setIsTtsPlaying(false)}
                onPlay={() => setIsTtsPlaying(true)}
              />
            ) : (
              <p className="empty">
                Submit text to stream speech. PCM starts playing as chunks arrive.
              </p>
            )}
            {ttsResult && <pre className="rawPreview">{JSON.stringify(ttsResult, null, 2)}</pre>}
          </div>
        </section>
      ) : activeTab === "clone" ? (
        <section className="cloneMode">
          <div className="panel cloneControls">
            <label>
              <span>Clone id</span>
              <input
                value={cloneId}
                onChange={(event) => setCloneId(event.target.value)}
                placeholder="my_voice"
              />
            </label>

            <label>
              <span>Language</span>
              <select
                value={cloneLanguage}
                onChange={(event) => setCloneLanguage(event.target.value)}
              >
                <option value="english">english</option>
                {ttsLanguages.filter((language) => language !== "english").map((language) => (
                  <option key={language} value={language}>
                    {language}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Reference audio</span>
              <input
                accept="audio/wav,audio/mpeg,audio/mp3,audio/ogg,.wav,.mp3,.ogg"
                type="file"
                onChange={(event) => setCloneFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <div className="buttonRow">
              <button onClick={createVoiceClone} disabled={isCloningVoice}>
                {isCloningVoice ? "Creating" : "Create clone"}
              </button>
              <button className="secondary" onClick={loadTtsVoices}>
                Refresh voices
              </button>
            </div>
            {cloneError && <p className="error">{cloneError}</p>}
          </div>

          <div className="panel cloneControls deleteCloneControls">
            <label>
              <span>Delete clone id</span>
              <input
                value={deleteCloneId}
                onChange={(event) => setDeleteCloneId(event.target.value)}
                placeholder="my_voice"
              />
            </label>
            <label>
              <span>Saved clones</span>
              <select
                value={deleteCloneId}
                onChange={(event) => setDeleteCloneId(event.target.value)}
              >
                <option value="">Select a clone</option>
                {clonedVoices.map((voice) => (
                  <option key={`delete-${voice.id}`} value={voice.id}>
                    {voice.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="buttonRow">
              <button
                className="secondary danger"
                onClick={deleteVoiceClone}
                disabled={isDeletingClone}
              >
                {isDeletingClone ? "Deleting" : "Delete clone"}
              </button>
            </div>
          </div>

          <div className="panel transcript fileResult">
            <div className="sectionHeader">
              <div>
                <h2>Clone Result</h2>
                <span>{cloneStatus}</span>
              </div>
            </div>
            <p className={cloneResult ? "" : "empty"}>
              {cloneResult
                ? `Voice ${cloneResult.clone_id ?? cloneResult.result?.clone_id ?? cloneId ?? "clone"} is ready to use by clone id.`
                : "Create a clone from a short reference audio file, then use its clone id as the TTS voice."}
            </p>
            {cloneResult && <pre className="rawPreview">{JSON.stringify(cloneResult, null, 2)}</pre>}
          </div>
        </section>
      ) : (
        <section className="chatMode">
          <div className="panel chatControls">
            <label>
              <span>Audio input</span>
              <select
                value={selectedAudioInputId}
                onChange={(event) => setSelectedAudioInputId(event.target.value)}
                disabled={isVoiceChatActive}
              >
                <option value="">Default microphone</option>
                {audioInputs.map((input) => (
                  <option key={input.deviceId} value={input.deviceId}>
                    {input.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Model</span>
              <div className="inlineField">
                <select
                  value={chatModel}
                  onChange={(event) => setChatModel(event.target.value)}
                  disabled={isChatModelsLoading && !chatModels.length}
                >
                  {!chatModels.length && (
                    <option value={chatModel || DEFAULT_CHAT_MODEL}>
                      {chatModel || DEFAULT_CHAT_MODEL}
                    </option>
                  )}
                  {chatModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                <button
                  className="secondary"
                  onClick={loadChatModels}
                  disabled={isChatModelsLoading}
                >
                  {isChatModelsLoading ? "Loading" : "Refresh"}
                </button>
              </div>
              {chatModelsError && <span className="fieldError">{chatModelsError}</span>}
            </label>

            <label>
              <span>Max tokens</span>
              <input
                type="number"
                min="16"
                max="2048"
                step="8"
                value={chatMaxTokens}
                onChange={(event) => setChatMaxTokens(event.target.value)}
                placeholder="No cap"
              />
            </label>

            <label>
              <span>Context window</span>
              <input
                type="number"
                min="512"
                max="32768"
                step="512"
                value={chatContextWindow}
                onChange={(event) => setChatContextWindow(event.target.value)}
                placeholder="Default"
              />
            </label>

            <label>
              <span>Turn mode</span>
              <select
                value={chatTurnMode}
                onChange={(event) => setChatTurnMode(event.target.value)}
              >
                <option value="balanced">Balanced</option>
                <option value="fast">Fast</option>
              </select>
            </label>

            <label>
              <span>Voice</span>
              <select
                value={ttsVoice}
                onChange={(event) => setTtsVoice(event.target.value)}
              >
                {renderVoiceOptions()}
              </select>
            </label>

            <div className="roomToneControl">
              <label className="checkboxLabel">
                <input
                  type="checkbox"
                  checked={isChatRoomToneEnabled}
                  onChange={(event) => setIsChatRoomToneEnabled(event.target.checked)}
                />
                <span>Room tone mask</span>
              </label>
              <label>
                <span>Mask volume {Number(chatRoomToneVolume).toFixed(1)}</span>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={chatRoomToneVolume}
                  onChange={(event) => setChatRoomToneVolume(event.target.value)}
                  disabled={!isChatRoomToneEnabled}
                />
              </label>
            </div>

            <label className="textAreaLabel">
              <span>System prompt</span>
              <textarea
                value={chatSystemPrompt}
                onChange={(event) => setChatSystemPrompt(event.target.value)}
                rows={3}
              />
            </label>

            <div className="buttonRow">
              <button onClick={startVoiceChat} disabled={isVoiceChatActive}>
                Start voice chat
              </button>
              <button
                className="secondary"
                onClick={stopVoiceChat}
                disabled={!isVoiceChatActive && !isChatResponding && !isChatSpeaking}
              >
                Stop
              </button>
              <button className="secondary" onClick={clearVoiceChat}>
                Clear
              </button>
              <button className="secondary" onClick={exportVoiceChatJson} disabled={!chatMessages.length}>
                Export JSON
              </button>
            </div>
            {chatError && <p className="error">{chatError}</p>}
          </div>

          <section className="chatWorkspace">
            <div className="panel chatConversation">
              <div className="sectionHeader">
                <div>
                  <h2>Conversation</h2>
                  <span>{chatMessages.length} messages</span>
                </div>
              </div>
              <div className="chatMessages">
                {chatMessages.map((message) => (
                  <article key={message.id} className={`chatMessage ${message.role}`}>
                    <time>{message.role}</time>
                    <p>{message.text || "..."}</p>
                  </article>
                ))}
                {!chatMessages.length && (
                  <p className="empty">Start voice chat, speak, then pause for VAD to send the turn.</p>
                )}
              </div>
            </div>

            <div className="panel chatSide">
              <div className="sectionHeader">
                <div>
                  <h2>Live Turn</h2>
                  <span>{chatVadStatus}</span>
                </div>
              </div>
              <p className={chatPartialTranscript ? "" : "empty"}>
                {chatPartialTranscript || "VAD partial speech will appear here while you talk."}
              </p>
              <div className="chatMeters">
                <span className={isVoiceChatActive ? "active" : ""}>ASR</span>
                <span className={isChatResponding ? "active" : ""}>LLM</span>
                <span className={isChatSpeaking ? "active" : ""}>TTS</span>
              </div>
              <div className="latencyList">
                <div className="sectionHeader compactHeader">
                  <div>
                    <h2>Current Turn</h2>
                    <span>
                      {chatTurnMode} · {chatLatencyEvents.length} steps
                    </span>
                  </div>
                </div>
                {chatTimingRows.length > 0 && (
                  <div className="latencySummary" aria-label="Chat latency breakdown">
                    <div className="latencySlowest">
                      <span>Slowest model leg</span>
                      <strong>
                        {slowestChatStage
                          ? `${slowestChatStage.label} ${formatLatency(slowestChatStage.valueMs)}`
                          : "Waiting for measurements"}
                      </strong>
                    </div>
                    <div className="latencyGrid">
                      {chatTimingRows.map((row) => {
                        const isSlowest = slowestChatStage?.id === row.id;
                        const isPending = row.valueMs == null;
                        return (
                          <article
                            key={row.id}
                            className={`latencyMetric${isSlowest ? " slowest" : ""}${
                              isPending ? " pending" : ""
                            }`}
                          >
                            <span>{row.label}</span>
                            <strong>{isPending ? "..." : formatLatency(row.valueMs)}</strong>
                            <small>{row.detail}</small>
                          </article>
                        );
                      })}
                    </div>
                  </div>
                )}
                <div className="sectionHeader compactHeader timelineHeader">
                  <div>
                    <h2>Debug Timeline</h2>
                    <span>{chatLatencyEvents.length} events</span>
                  </div>
                </div>
                <div className="latencyScroll" aria-label="Latency events">
                  {chatLatencyEvents.map((event) => (
                    <article key={event.id} className="latencyEvent">
                      <time>{formatLatency(event.elapsedMs)}</time>
                      <p>
                        <strong>{event.label}</strong>
                        {event.detail ? ` · ${event.detail}` : ""}
                      </p>
                    </article>
                  ))}
                  {!chatLatencyEvents.length && (
                    <p className="empty">Timing events will appear here during a turn.</p>
                  )}
                </div>
              </div>
            </div>
          </section>
        </section>
      )}
    </main>
  );
}
