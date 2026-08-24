/**
 * AADS Frontend API Client
 * Interfaces with the FastAPI backend at http://localhost:8000
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('File upload failed');
  return res.json();
}

export async function createSampleDataset() {
  const res = await fetch(`${API_BASE}/api/sample-data`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to generate sample data');
  return res.json();
}

export async function previewDataset(path, rows = 6) {
  const res = await fetch(`${API_BASE}/api/preview?path=${encodeURIComponent(path)}&rows=${rows}`);
  if (!res.ok) throw new Error('Failed to preview dataset');
  return res.json();
}

export async function launchPipeline(payload) {
  const res = await fetch(`${API_BASE}/api/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to launch pipeline');
  return res.json();
}

export function subscribeToPipelineStream(runId, onMessage, onError, onComplete) {
  const eventSource = new EventSource(`${API_BASE}/api/pipeline/${runId}/stream`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        onMessage(data.message);
      } else if (data.type === 'complete') {
        eventSource.close();
        if (onComplete) onComplete(data.message);
      } else if (data.type === 'error') {
        eventSource.close();
        if (onError) onError(data.message);
      }
    } catch (e) {
      console.error('Failed to parse SSE event', e);
    }
  };

  eventSource.onerror = (err) => {
    console.error('SSE Error:', err);
    eventSource.close();
    if (onError) onError('Stream connection error');
  };

  return () => eventSource.close();
}

export async function getPipelineResult(runId) {
  const res = await fetch(`${API_BASE}/api/pipeline/${runId}/result`);
  if (!res.ok) throw new Error('Failed to fetch pipeline result');
  return res.json();
}

export async function getLatestPipelineResult() {
  const res = await fetch(`${API_BASE}/api/pipeline/latest`);
  if (!res.ok) throw new Error('Failed to fetch latest result');
  return res.json();
}

export async function listPipelineFiles(runId) {
  const res = await fetch(`${API_BASE}/api/pipeline/${runId}/files`);
  if (!res.ok) throw new Error('Failed to list files');
  return res.json();
}

export function getFileDownloadUrl(runId, filePath) {
  return `${API_BASE}/api/pipeline/${runId}/files/${filePath}`;
}

export function getZipDownloadUrl(runId) {
  return `${API_BASE}/api/pipeline/${runId}/zip`;
}

export async function listVisualizations(runId) {
  const res = await fetch(`${API_BASE}/api/pipeline/${runId}/visualizations`);
  if (!res.ok) throw new Error('Failed to fetch visualizations');
  return res.json();
}

export async function getSettings() {
  const res = await fetch(`${API_BASE}/api/settings`);
  if (!res.ok) throw new Error('Failed to load settings');
  return res.json();
}

export async function saveSettings(settings) {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  });
  if (!res.ok) throw new Error('Failed to save settings');
  return res.json();
}

export async function testConnection(provider, model, apiKey) {
  const res = await fetch(`${API_BASE}/api/test-connection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model, api_key: apiKey }),
  });
  if (!res.ok) throw new Error('Connection test failed');
  return res.json();
}

export async function fetchProviderModels(provider, apiKey) {
  const res = await fetch(`${API_BASE}/api/fetch-models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}
