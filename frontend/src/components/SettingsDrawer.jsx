import React, { useState, useEffect } from 'react';
import { getSettings, saveSettings, testConnection, fetchProviderModels } from '../api';
import { SearchableSelect } from './SearchableSelect';

const DEFAULT_PROVIDER_MODELS = {
  openrouter: [
    'google/gemini-2.0-flash-001',
    'meta-llama/llama-3.3-70b-instruct',
    'openai/gpt-4o-mini',
    'anthropic/claude-3.5-sonnet',
    'deepseek/deepseek-r1',
  ],
  nvidia: [
    'meta/llama-3.3-70b-instruct',
    'deepseek-ai/deepseek-r1',
    'nvidia/llama-3.1-nemotron-70b-instruct',
    'mistralai/mistral-large-2-instruct',
    'meta/llama-3.1-8b-instruct',
    'nvidia/nemotron-4-340b-instruct',
  ],
  google: [
    'gemini-2.0-flash',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
  ],
  openai: [
    'gpt-4o-mini',
    'gpt-4o',
    'o3-mini',
  ],
  anthropic: [
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
  ],
  ollama: [
    'llama3.2',
    'llama3.1',
    'mistral',
  ],
  custom: [
    'custom-model',
  ],
};

const PROVIDER_OPTIONS = [
  { id: 'openrouter', label: 'OpenRouter (Multi-Model)' },
  { id: 'nvidia', label: 'NVIDIA NIM (High Performance)' },
  { id: 'google', label: 'Google Gemini' },
  { id: 'openai', label: 'OpenAI' },
  { id: 'anthropic', label: 'Anthropic Claude' },
  { id: 'ollama', label: 'Ollama (Local LLM)' },
  { id: 'custom', label: 'Custom (OpenAI-Compatible Endpoint)' },
];

export function SettingsDrawer({ isOpen, onClose, onSettingsSaved }) {
  const [settings, setSettings] = useState({
    execution_mode: 'local',
    selected_provider: 'openrouter',
    api_keys: {},
    provider_models_cache: {},
    selected_model: '',
    custom_base_url: '',
    custom_provider_name: '',
    autonomy: 'fully_autonomous',
    engine: 'pandas',
    random_seed: 42,
    storage_dir: '',
  });

  const [availableModels, setAvailableModels] = useState([]);
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadSettingsData();
    }
  }, [isOpen]);

  const loadSettingsData = async () => {
    try {
      const res = await getSettings();
      const s = res.settings || {};
      const provider = s.selected_provider || 'openrouter';
      const modelsMap = s.provider_models_cache || {};
      const defaultModels = res.default_models || DEFAULT_PROVIDER_MODELS;
      const models = modelsMap[provider] || defaultModels[provider] || DEFAULT_PROVIDER_MODELS[provider] || [];

      setSettings((prev) => ({ ...prev, ...s }));
      setAvailableModels(models);
    } catch (err) {
      console.error('Failed to load settings:', err);
    }
  };

  const handleProviderChange = (provider) => {
    const modelsMap = settings.provider_models_cache || {};
    const models = modelsMap[provider] || DEFAULT_PROVIDER_MODELS[provider] || [];
    setSettings((prev) => ({
      ...prev,
      selected_provider: provider,
      selected_model: models[0] || '',
    }));
    setAvailableModels(models);
    setTestResult(null);
  };

  const handleApiKeyChange = (key) => {
    setSettings((prev) => ({
      ...prev,
      api_keys: {
        ...(prev.api_keys || {}),
        [prev.selected_provider]: key,
      },
    }));
  };

  const handleFetchModels = async () => {
    setFetching(true);
    setTestResult(null);
    try {
      const apiKey = settings.api_keys?.[settings.selected_provider] || '';
      const baseUrl = settings.selected_provider === 'custom' ? settings.custom_base_url : null;
      const res = await fetchProviderModels(settings.selected_provider, apiKey, baseUrl);
      if (res.models && res.models.length > 0) {
        setAvailableModels(res.models);
        const updatedMap = { ...(settings.provider_models_cache || {}), [settings.selected_provider]: res.models };
        setSettings((prev) => ({
          ...prev,
          provider_models_cache: updatedMap,
          selected_model: res.models[0],
        }));
        setTestResult({ success: true, message: `Fetched ${res.models.length} models!` });
      }
    } catch (err) {
      setTestResult({ success: false, message: err.message || 'Failed to fetch models' });
    } finally {
      setFetching(false);
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const apiKey = settings.api_keys?.[settings.selected_provider] || '';
      const model = settings.selected_model || availableModels[0] || 'default';
      const baseUrl = settings.selected_provider === 'custom' ? settings.custom_base_url : null;
      const res = await testConnection(settings.selected_provider, model, apiKey, baseUrl);
      setTestResult(res);
    } catch (err) {
      setTestResult({ success: false, message: err.message || 'Connection test failed' });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveSettings(settings);
      if (onSettingsSaved) onSettingsSaved(settings);
      onClose();
    } catch (err) {
      alert('Failed to save settings: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  const currentApiKey = settings.api_keys?.[settings.selected_provider] || '';

  return (
    <>
      <div className={`settings-overlay ${isOpen ? 'open' : ''}`} onClick={onClose} />
      <div className={`settings-drawer ${isOpen ? 'open' : ''}`}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
          <h3 style={{ margin: 0 }}>⚙️ Agent & Model Settings</h3>
          <button className="btn btn-ghost" onClick={onClose} style={{ padding: '4px 10px' }}>✕</button>
        </div>

        {/* Execution Mode */}
        <div className="settings-group">
          <div className="settings-group-title">Execution Mode</div>
          <div className="settings-radio-group">
            <div
              className={`settings-radio ${settings.execution_mode === 'local' ? 'active' : ''}`}
              onClick={() => setSettings((prev) => ({ ...prev, execution_mode: 'local' }))}
            >
              💻 Local (Deterministic)
            </div>
            <div
              className={`settings-radio ${settings.execution_mode === 'ai' ? 'active' : ''}`}
              onClick={() => setSettings((prev) => ({ ...prev, execution_mode: 'ai' }))}
            >
              ✨ AI-Powered (LLM)
            </div>
          </div>
        </div>

        {/* AI Provider Settings (conditional) */}
        {settings.execution_mode === 'ai' && (
          <div className="settings-group">
            <div className="settings-group-title">🤖 LLM Provider Configuration</div>

            <div className="settings-field">
              <label className="settings-label">Provider</label>
              <select
                className="settings-select"
                value={settings.selected_provider}
                onChange={(e) => handleProviderChange(e.target.value)}
              >
                {PROVIDER_OPTIONS.map((p) => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>
            </div>

            {/* Custom Base URL (if custom provider selected) */}
            {settings.selected_provider === 'custom' && (
              <>
                <div className="settings-field">
                  <label className="settings-label">Custom API Base URL</label>
                  <input
                    type="text"
                    className="settings-input"
                    placeholder="e.g. https://api.together.xyz/v1, http://localhost:1234/v1"
                    value={settings.custom_base_url || ''}
                    onChange={(e) => setSettings((prev) => ({ ...prev, custom_base_url: e.target.value }))}
                  />
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Works with any OpenAI-compatible server (vLLM, LM Studio, Together AI, DeepSeek API, Mistral).
                  </div>
                </div>

                <div className="settings-field">
                  <label className="settings-label">Custom Provider Label (Optional)</label>
                  <input
                    type="text"
                    className="settings-input"
                    placeholder="e.g. Together AI / DeepSeek"
                    value={settings.custom_provider_name || ''}
                    onChange={(e) => setSettings((prev) => ({ ...prev, custom_provider_name: e.target.value }))}
                  />
                </div>
              </>
            )}

            <div className="settings-field">
              <label className="settings-label">
                {settings.selected_provider === 'custom' ? 'API Key (Optional for local servers)' : `${settings.selected_provider.toUpperCase()} API Key`}
              </label>
              <input
                type="password"
                className="settings-input"
                placeholder={settings.selected_provider === 'nvidia' ? 'nvapi-...' : 'sk-...'}
                value={currentApiKey}
                onChange={(e) => handleApiKeyChange(e.target.value)}
              />
            </div>

            <div className="settings-btn-row">
              <button
                className="btn btn-ghost"
                style={{ flex: 1, fontSize: 'var(--text-xs)' }}
                onClick={handleFetchModels}
                disabled={fetching}
              >
                {fetching ? 'Fetching...' : '🔄 Fetch Models'}
              </button>
              <button
                className="btn btn-ghost"
                style={{ flex: 1, fontSize: 'var(--text-xs)' }}
                onClick={handleTestConnection}
                disabled={testing}
              >
                {testing ? 'Testing...' : '⚡ Test API'}
              </button>
            </div>

            {testResult && (
              <div
                style={{
                  marginTop: '10px',
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 'var(--text-xs)',
                  fontFamily: 'var(--font-mono)',
                  background: testResult.success ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                  color: testResult.success ? '#6ee7b7' : '#fca5a5',
                  border: `1px solid ${testResult.success ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)'}`,
                }}
              >
                {testResult.success ? '✓' : '⚠️'} {testResult.message}
              </div>
            )}

            <div className="settings-field" style={{ marginTop: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label className="settings-label" style={{ margin: 0 }}>Active Model Identifier</label>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Type or click dropdown list</span>
              </div>
              <SearchableSelect
                value={settings.selected_model || ''}
                onChange={(val) => setSettings((prev) => ({ ...prev, selected_model: val }))}
                options={availableModels}
                placeholder="Type to filter or select model..."
                direction="up"
              />

              {/* Quick Model Presets for OpenRouter / Providers */}
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '8px' }}>
                {(settings.selected_provider === 'openrouter'
                  ? ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'deepseek/deepseek-chat', 'google/gemini-2.0-flash-001', 'poolside/laguna-s-2.1:free']
                  : (availableModels.slice(0, 4))
                ).map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    className="btn btn-ghost"
                    style={{
                      padding: '2px 8px',
                      fontSize: '10px',
                      borderRadius: '4px',
                      border: '1px solid var(--border-subtle)',
                      background: settings.selected_model === preset ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                      color: settings.selected_model === preset ? '#93c5fd' : 'var(--text-secondary)',
                    }}
                    onClick={() => setSettings((prev) => ({ ...prev, selected_model: preset }))}
                  >
                    {preset.split('/')[1] || preset}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Engine & Runtime */}
        <div className="settings-group">
          <div className="settings-group-title">🛠️ Runtime Engine</div>

          <div className="settings-field">
            <label className="settings-label">Autonomy Level</label>
            <select
              className="settings-select"
              value={settings.autonomy}
              onChange={(e) => setSettings((prev) => ({ ...prev, autonomy: e.target.value }))}
            >
              <option value="fully_autonomous">Fully Autonomous (Standard)</option>
              <option value="semi_autonomous">Semi-Autonomous (Interactive checkpoints)</option>
              <option value="manual_approval">Manual Approval</option>
            </select>
          </div>

          <div className="settings-field">
            <label className="settings-label">Data Processing Engine</label>
            <select
              className="settings-select"
              value={settings.engine}
              onChange={(e) => setSettings((prev) => ({ ...prev, engine: e.target.value }))}
            >
              <option value="pandas">Pandas (Standard)</option>
              <option value="polars">Polars (High Performance)</option>
              <option value="duckdb">DuckDB (Analytical SQL)</option>
            </select>
          </div>

          <div className="settings-field">
            <label className="settings-label">Random Seed</label>
            <input
              type="number"
              className="settings-input"
              value={settings.random_seed}
              onChange={(e) => setSettings((prev) => ({ ...prev, random_seed: parseInt(e.target.value, 10) || 42 }))}
            />
          </div>
        </div>

        {/* Action Button */}
        <button
          className="btn btn-primary"
          style={{ width: '100%', marginTop: '16px' }}
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : '💾 Save Settings'}
        </button>
      </div>
    </>
  );
}
