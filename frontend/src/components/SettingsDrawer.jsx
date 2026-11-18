import React, { useState, useEffect } from 'react';
import { getSettings, saveSettings, testConnection, fetchProviderModels } from '../api';
import { SearchableSelect } from './SearchableSelect';
import { Icon } from './Icon';

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
  google: ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'],
  openai: ['gpt-4o-mini', 'gpt-4o', 'o3-mini'],
  anthropic: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022'],
  ollama: ['llama3.2', 'llama3.1', 'mistral'],
  custom: ['custom-model'],
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
    if (isOpen) loadSettingsData();
  }, [isOpen]);

  const loadSettingsData = async () => {
    try {
      const res = await getSettings();
      const s = res.settings || {};
      const provider = s.selected_provider || 'openrouter';
      const modelsMap = s.provider_models_cache || {};
      const defaultModels = res.default_models || DEFAULT_PROVIDER_MODELS;
      let models = modelsMap[provider] || defaultModels[provider] || DEFAULT_PROVIDER_MODELS[provider] || [];

      setSettings((prev) => ({ ...prev, ...s }));
      setAvailableModels(models);

      // If the backend has no cached list for this provider yet, pull one
      // from /api/fetch-models so the combobox reflects the backend.
      if (models.length === 0) {
        try {
          const apiKey = s.api_keys?.[provider] || '';
          const baseUrl = provider === 'custom' ? s.custom_base_url : null;
          const fetched = await fetchProviderModels(provider, apiKey, baseUrl);
          if (fetched.models && fetched.models.length > 0) {
            setAvailableModels(fetched.models);
            setSettings((prev) => ({
              ...prev,
              provider_models_cache: { ...(prev.provider_models_cache || {}), [provider]: fetched.models },
            }));
          }
        } catch (_) {
          /* silent — the Fetch models button remains the manual path */
        }
      }
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
      api_keys: { ...(prev.api_keys || {}), [prev.selected_provider]: key },
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
        setTestResult({ success: true, message: `Fetched ${res.models.length} models from provider.` });
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
      setTestResult({ success: false, message: 'Failed to save settings: ' + err.message });
    } finally {
      setSaving(false);
    }
  };

  const currentApiKey = settings.api_keys?.[settings.selected_provider] || '';
  const isCustom = settings.selected_provider === 'custom';

  return (
    <>
      <div className={`drawer-overlay ${isOpen ? 'open' : ''}`} onClick={onClose} aria-hidden="true" />
      <div
        className={`drawer ${isOpen ? 'open' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label="Agent and model settings"
      >
        <div className="drawer-head">
          <div className="drawer-title">
            <Icon name="sliders" size={20} />
            Agent &amp; Model Settings
          </div>
          <button className="btn btn-ghost btn-icon-only" onClick={onClose} aria-label="Close settings">
            <Icon name="x" size={17} />
          </button>
        </div>

        <div className="drawer-body">
          {/* Execution mode */}
          <div className="settings-group">
            <div className="settings-group-title">
              <Icon name="cpu" size={14} />
              Execution Mode
            </div>
            <div className="segmented">
              <button
                type="button"
                className={`segmented-btn ${settings.execution_mode === 'local' ? 'active' : ''}`}
                onClick={() => setSettings((prev) => ({ ...prev, execution_mode: 'local' }))}
              >
                <Icon name="zap" size={15} />
                Local · Deterministic
              </button>
              <button
                type="button"
                className={`segmented-btn ${settings.execution_mode === 'ai' ? 'active' : ''}`}
                onClick={() => setSettings((prev) => ({ ...prev, execution_mode: 'ai' }))}
              >
                <Icon name="sparkles" size={15} />
                AI-Powered · LLM
              </button>
            </div>
          </div>

          {/* LLM provider (conditional) */}
          {settings.execution_mode === 'ai' && (
            <div className="settings-group">
              <div className="settings-group-title">
                <Icon name="sparkles" size={14} />
                LLM Provider
              </div>

              <div>
                <label className="field-label" htmlFor="provider-select">Provider</label>
                <select
                  id="provider-select"
                  className="select"
                  value={settings.selected_provider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                >
                  {PROVIDER_OPTIONS.map((p) => (
                    <option key={p.id} value={p.id}>{p.label}</option>
                  ))}
                </select>
              </div>

              {isCustom && (
                <>
                  <div>
                    <label className="field-label" htmlFor="base-url">Custom API base URL</label>
                    <input
                      id="base-url"
                      type="text"
                      className="input input-mono"
                      placeholder="https://api.together.xyz/v1 · http://localhost:1234/v1"
                      value={settings.custom_base_url || ''}
                      onChange={(e) => setSettings((prev) => ({ ...prev, custom_base_url: e.target.value }))}
                    />
                    <div className="field-hint">
                      Works with any OpenAI-compatible server: vLLM, LM Studio, Together AI, DeepSeek API, Mistral.
                    </div>
                  </div>

                  <div>
                    <label className="field-label" htmlFor="provider-label">Custom provider label (optional)</label>
                    <input
                      id="provider-label"
                      type="text"
                      className="input"
                      placeholder="e.g. Together AI / DeepSeek"
                      value={settings.custom_provider_name || ''}
                      onChange={(e) => setSettings((prev) => ({ ...prev, custom_provider_name: e.target.value }))}
                    />
                  </div>
                </>
              )}

              <div>
                <label className="field-label" htmlFor="api-key">
                  {isCustom ? 'API key (optional for local servers)' : `${settings.selected_provider.toUpperCase()} API key`}
                </label>
                <input
                  id="api-key"
                  type="password"
                  className="input input-mono"
                  placeholder={settings.selected_provider === 'nvidia' ? 'nvapi-…' : 'sk-…'}
                  value={currentApiKey}
                  onChange={(e) => handleApiKeyChange(e.target.value)}
                  autoComplete="off"
                />
              </div>

              <div className="settings-btn-row">
                <button className="btn btn-secondary" onClick={handleFetchModels} disabled={fetching}>
                  <Icon name="refresh" size={15} />
                  {fetching ? 'Fetching…' : 'Fetch models'}
                </button>
                <button className="btn btn-secondary" onClick={handleTestConnection} disabled={testing}>
                  <Icon name="plug" size={15} />
                  {testing ? 'Testing…' : 'Test API'}
                </button>
              </div>

              {testResult && (
                <div className={`notice ${testResult.success ? 'notice-jade' : 'notice-alert'}`}>
                  <Icon name={testResult.success ? 'check-circle' : 'alert-triangle'} size={16} />
                  <span>{testResult.message}</span>
                </div>
              )}

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 7 }}>
                  <label className="field-label" style={{ margin: 0 }} htmlFor="model-combo">Active model identifier</label>
                  <span className="field-hint" style={{ margin: 0 }}>type to filter · ↑↓ to navigate</span>
                </div>
                <SearchableSelect
                  value={settings.selected_model || ''}
                  onChange={(val) => setSettings((prev) => ({ ...prev, selected_model: val }))}
                  options={availableModels}
                  placeholder="Type to filter or select a model…"
                  direction="down"
                />

                <div className="preset-row">
                  {(settings.selected_provider === 'openrouter'
                    ? ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'deepseek/deepseek-chat', 'google/gemini-2.0-flash-001']
                    : availableModels.slice(0, 4)
                  ).map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      className={`preset-chip ${settings.selected_model === preset ? 'active' : ''}`}
                      onClick={() => setSettings((prev) => ({ ...prev, selected_model: preset }))}
                    >
                      {preset.split('/')[1] || preset}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Runtime engine */}
          <div className="settings-group">
            <div className="settings-group-title">
              <Icon name="layers" size={14} />
              Runtime Engine
            </div>

            <div>
              <label className="field-label" htmlFor="autonomy">Autonomy level</label>
              <select
                id="autonomy"
                className="select"
                value={settings.autonomy}
                onChange={(e) => setSettings((prev) => ({ ...prev, autonomy: e.target.value }))}
              >
                <option value="fully_autonomous">Fully Autonomous (Standard)</option>
                <option value="semi_autonomous">Semi-Autonomous (Interactive checkpoints)</option>
                <option value="manual_approval">Manual Approval</option>
              </select>
            </div>

            <div>
              <label className="field-label" htmlFor="engine">Data processing engine</label>
              <select
                id="engine"
                className="select"
                value={settings.engine}
                onChange={(e) => setSettings((prev) => ({ ...prev, engine: e.target.value }))}
              >
                <option value="pandas">Pandas (Standard)</option>
                <option value="polars">Polars (High Performance)</option>
                <option value="duckdb">DuckDB (Analytical SQL)</option>
              </select>
            </div>

            <div>
              <label className="field-label" htmlFor="seed">Random seed</label>
              <input
                id="seed"
                type="number"
                className="input input-mono"
                value={settings.random_seed}
                onChange={(e) => setSettings((prev) => ({ ...prev, random_seed: parseInt(e.target.value, 10) || 42 }))}
              />
            </div>
          </div>
        </div>

        <div className="drawer-foot">
          <button className="btn btn-primary btn-pill" style={{ width: '100%' }} onClick={handleSave} disabled={saving}>
            <Icon name="save" size={16} />
            {saving ? 'Saving…' : 'Save settings'}
          </button>
        </div>
      </div>
    </>
  );
}
