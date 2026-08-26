/**
 * AUDAS Frontend API Client
 * Interfaces with the FastAPI backend at http://localhost:8000
 */

export const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export function uploadDataset(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    if (onProgress && xhr.upload) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          onProgress({ percent: pct, loaded: e.loaded, total: e.total });
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch (err) {
          reject(new Error('Failed to parse upload response'));
        }
      } else {
        let msg = 'File upload failed';
        try {
          const errData = JSON.parse(xhr.responseText);
          if (errData.detail) msg = errData.detail;
        } catch (_) {}
        reject(new Error(msg));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during file upload'));
    xhr.ontimeout = () => reject(new Error('File upload timed out'));

    xhr.open('POST', `${API_BASE}/api/upload`);
    xhr.send(formData);
  });
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

  eventSource.onerror = async (err) => {
    console.warn('SSE stream notice:', err);
    eventSource.close();
    
    // Check if backend already finished or has an error before showing fatal error
    try {
      const res = await getPipelineResult(runId);
      if (res && res.status === 'completed' && res.result) {
        if (onComplete) onComplete('Pipeline completed successfully!');
        return;
      }
      if (res && res.status === 'error') {
        if (onError) onError(res.error || 'Pipeline execution failed');
        return;
      }
    } catch (_) {}

    // Polling retry fallback: check result after 2 seconds
    setTimeout(async () => {
      try {
        const retryRes = await getPipelineResult(runId);
        if (retryRes && retryRes.status === 'completed' && retryRes.result) {
          if (onComplete) onComplete('Pipeline completed successfully!');
          return;
        }
      } catch (_) {}
      if (onError) onError('Pipeline connection closed. Please check status.');
    }, 2000);
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

export function getFileDownloadUrl(runId, filePath, isDownload = true) {
  const query = isDownload ? '?download=true' : '?download=false';
  return `${API_BASE}/api/pipeline/${runId}/files/${filePath}${query}`;
}

export function getZipDownloadUrl(runId) {
  return `${API_BASE}/api/pipeline/${runId}/zip`;
}

export function getModelDownloadUrl(runId, rank) {
  return `${API_BASE}/api/pipeline/${runId}/models/${rank}/download`;
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

export async function testConnection(provider, model, apiKey, baseUrl = null) {
  const res = await fetch(`${API_BASE}/api/test-connection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model, api_key: apiKey, base_url: baseUrl }),
  });
  if (!res.ok) throw new Error('Connection test failed');
  return res.json();
}

export async function fetchProviderModels(provider, apiKey, baseUrl = null) {
  const res = await fetch(`${API_BASE}/api/fetch-models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, api_key: apiKey, base_url: baseUrl }),
  });
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}
