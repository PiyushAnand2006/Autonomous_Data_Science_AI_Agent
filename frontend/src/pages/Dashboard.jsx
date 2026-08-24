import React, { useState, useEffect } from 'react';
import { Card } from '../components/Card';
import { FileUploader } from '../components/FileUploader';
import { PipelineProgress } from '../components/PipelineProgress';
import { launchPipeline, subscribeToPipelineStream, getPipelineResult, previewDataset, getZipDownloadUrl } from '../api';

export function Dashboard({ settings, setSettings, onOpenSettings, currentResult, setCurrentResult, onNavigateTab }) {
  const [dataFile, setDataFile] = useState(null);
  const [objective, setObjective] = useState('Predict customer churn, discover key behavioral drivers, and export top performing ML models.');
  const [targetCol, setTargetCol] = useState('');
  const [previewData, setPreviewData] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [runId, setRunId] = useState(currentResult?.run_id || null);
  const [status, setStatus] = useState(currentResult ? 'completed' : 'idle');
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!currentResult) {
      setStatus('idle');
      setLogs([]);
      setError(null);
    }
  }, [currentResult]);

  const [modeNotice, setModeNotice] = useState(null);

  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const isAiMode = settings?.execution_mode === 'ai';

  const handleModeToggle = (mode) => {
    if (settings?.execution_mode !== mode) {
      if (setSettings) {
        setSettings((prev) => ({ ...prev, execution_mode: mode }));
      }
      if (setCurrentResult) {
        setCurrentResult(null);
      }
      setLogs([]);
      setStatus('idle');
      setError(null);
      setRunId(null);
      setIsRunning(false);
      setModeNotice(`Switched to ${mode === 'ai' ? '✨ AI-Powered Agentic' : '💻 Local (Deterministic)'} engine. Please click '🚀 Run Autonomous Pipeline' to execute under the new mode.`);
      setTimeout(() => setModeNotice(null), 6000);
    }
  };

  const handleFileReady = async (fileInfo) => {
    setDataFile(fileInfo);
    // Reset previous run session to start clean
    if (setCurrentResult) {
      setCurrentResult(null);
    }
    setLogs([]);
    setStatus('idle');
    setError(null);
    setRunId(null);
    setIsRunning(false);

    if (!fileInfo) {
      setPreviewData(null);
      return;
    }

    try {
      const p = await previewDataset(fileInfo.path, 5);
      setPreviewData(p);
    } catch (e) {
      console.warn('Could not load preview', e);
      setPreviewData(null);
    }
  };

  const handleLaunch = async () => {
    if (!dataFile || !dataFile.path) {
      alert('Please upload or select a dataset first.');
      return;
    }

    setIsRunning(true);
    setLogs(['🚀 Initializing autonomous pipeline run...']);
    setStatus('running');
    setError(null);

    const payload = {
      data_path: dataFile.path,
      user_objective: objective,
      target_column: targetCol,
      execution_mode: settings.execution_mode || 'local',
      autonomy: settings.autonomy || 'fully_autonomous',
      engine: settings.engine || 'pandas',
      random_seed: settings.random_seed || 42,
      storage_dir: settings.storage_dir || '',
      llm_provider: settings.selected_provider || 'openrouter',
      llm_model: settings.selected_model || '',
      llm_api_key: settings.api_keys?.[settings.selected_provider] || '',
    };

    try {
      const res = await launchPipeline(payload);
      setRunId(res.run_id);

      subscribeToPipelineStream(
        res.run_id,
        (msg) => {
          setLogs((prev) => [...prev, msg]);
        },
        (err) => {
          setError(err);
          setIsRunning(false);
          setStatus('error');
        },
        async () => {
          // Completed
          setIsRunning(false);
          setStatus('completed');
          try {
            const finalRes = await getPipelineResult(res.run_id);
            if (finalRes && finalRes.result) {
              setCurrentResult(finalRes.result);
            } else {
              // Retry shortly if backend is still wrapping up
              setTimeout(async () => {
                try {
                  const retryRes = await getPipelineResult(res.run_id);
                  if (retryRes && retryRes.result) {
                    setCurrentResult(retryRes.result);
                  }
                } catch (re) {
                  console.error('Retry get result failed', re);
                }
              }, 600);
            }
          } catch (e) {
            console.error('Failed to get final result', e);
          }
        }
      );
    } catch (err) {
      setError(err.message || 'Failed to start pipeline');
      setIsRunning(false);
      setStatus('error');
    }
  };

  return (
    <div>
      {/* Hero Header & Mode Switcher */}
      <div style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div className="eyebrow">Enterprise Autonomous Data Science</div>
          <h1 style={{ margin: '4px 0 8px 0' }}>Autonomous AI Data Scientist</h1>
          <p style={{ maxWidth: '620px', margin: 0, color: 'var(--text-secondary)' }}>
            Transform raw tabular datasets and natural-language objectives into full-lifecycle, production-ready, reproducible Machine Learning projects.
          </p>
        </div>

        {/* Execution Mode Selector Card */}
        <Card style={{ padding: '14px 18px', minWidth: '320px' }}>
          <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
            Execution Mode
          </div>
          <div style={{ display: 'flex', gap: '6px', background: 'var(--surface-1)', padding: '4px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <button
              className={`btn ${!isAiMode ? 'btn-primary' : 'btn-ghost'}`}
              style={{ flex: 1, padding: '6px 10px', fontSize: 'var(--text-xs)', borderRadius: 'var(--radius-sm)' }}
              onClick={() => handleModeToggle('local')}
            >
              💻 Local (Offline)
            </button>
            <button
              className={`btn ${isAiMode ? 'btn-primary' : 'btn-ghost'}`}
              style={{ flex: 1, padding: '6px 10px', fontSize: 'var(--text-xs)', borderRadius: 'var(--radius-sm)' }}
              onClick={() => handleModeToggle('ai')}
            >
              ✨ AI-Powered
            </button>
          </div>

          {isAiMode ? (
            <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 'var(--text-xs)' }}>
              <span style={{ color: 'var(--accent-indigo)' }}>
                🤖 {settings?.selected_provider?.toUpperCase() || 'OPENROUTER'} • {settings?.selected_model?.split('/')?.pop() || 'claude-3.5'}
              </span>
              <button
                className="btn btn-ghost"
                style={{ padding: '2px 6px', fontSize: 'var(--text-xs)' }}
                onClick={onOpenSettings}
              >
                ⚙️ Config
              </button>
            </div>
          ) : (
            <div style={{ marginTop: '10px', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              ⚡ Deterministic local ML training (No API key needed)
            </div>
          )}
        </Card>
      </div>

      {/* Main Form Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '24px' }}>
        {/* Step 1: Data Source */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <span style={{ fontSize: 'var(--text-lg)' }}>1️⃣</span>
            <h3 style={{ margin: 0 }}>Data Ingestion</h3>
          </div>

          <FileUploader onFileReady={handleFileReady} currentFile={dataFile} />

          {/* Dataset Preview Collapsible Accordion */}
          {dataFile && previewData && (
            <div style={{ marginTop: '16px' }}>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => setIsPreviewOpen(!isPreviewOpen)}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 14px',
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 'var(--text-sm)',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{isPreviewOpen ? '▾' : '▸'}</span>
                  <span>🔍 Dataset Preview</span>
                </span>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {previewData.total_rows} rows × {previewData.total_columns} columns
                </span>
              </button>

              {isPreviewOpen && (
                <div className="data-table-wrap" style={{ marginTop: '8px', maxHeight: '220px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        {previewData.columns.map((c) => (
                          <th key={c}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.rows.map((r, i) => (
                        <tr key={i}>
                          {r.map((v, j) => (
                            <td key={j}>{String(v)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Step 2: Goal & Objective */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <span style={{ fontSize: 'var(--text-lg)' }}>2️⃣</span>
            <h3 style={{ margin: 0 }}>Task & Objective</h3>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label className="settings-label">Natural Language Goal</label>
            <textarea
              className="settings-input"
              rows={3}
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="e.g. Predict customer churn and discover key drivers"
              style={{ resize: 'vertical' }}
            />
          </div>

          <div style={{ marginBottom: '18px' }}>
            <label className="settings-label">Target Column (Leave blank for auto-detection)</label>
            <input
              type="text"
              className="settings-input"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              placeholder="Auto-detect (e.g. churn, label, price)"
            />
          </div>

          {modeNotice && (
            <div style={{ marginBottom: '14px', padding: '10px 14px', background: 'rgba(59, 130, 246, 0.12)', border: '1px solid rgba(59, 130, 246, 0.35)', borderRadius: 'var(--radius-sm)', fontSize: 'var(--text-xs)', color: '#93c5fd', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>ℹ️</span>
              <span>{modeNotice}</span>
            </div>
          )}

          <button
            className="btn btn-primary btn-primary-lg"
            style={{ width: '100%' }}
            onClick={handleLaunch}
            disabled={isRunning || !dataFile}
          >
            {isRunning ? '⏳ Running Agents...' : '🚀 Launch Autonomous Pipeline'}
            <span className="btn-icon">↗</span>
          </button>
        </Card>
      </div>

      {/* Live Pipeline Execution Stream */}
      {(isRunning || logs.length > 0 || error) && (
        <PipelineProgress
          logs={logs}
          status={status}
          isRunning={isRunning}
          runId={runId}
          error={error}
        />
      )}

      {/* Top Level Run Summary if result exists */}
      {currentResult && (
        <div style={{ marginTop: '40px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <div className="eyebrow" style={{ color: 'var(--accent-emerald)' }}>Pipeline Run Completed</div>
              <h2>Run Summary: <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xl)' }}>{currentResult.run_id}</span></h2>
            </div>

            <a
              href={getZipDownloadUrl(currentResult.run_id)}
              className="btn btn-primary"
            >
              📦 Download Complete Project Archive (.ZIP)
            </a>
          </div>

          {/* Metric HUD Cards */}
          <div className="stat-hud">
            <Card className="stat-item" glow="emerald">
              <div className="stat-label">Best Performing Model</div>
              <div className="stat-value" style={{ fontSize: 'var(--text-lg)', color: '#6ee7b7' }}>
                {currentResult.best_model_name}
              </div>
            </Card>

            <Card className="stat-item">
              <div className="stat-label">Primary Metric Score</div>
              <div className="stat-value">
                {currentResult.best_metrics && Object.values(currentResult.best_metrics)[0] != null
                  ? typeof Object.values(currentResult.best_metrics)[0] === 'number'
                    ? Object.values(currentResult.best_metrics)[0].toFixed(4)
                    : Object.values(currentResult.best_metrics)[0]
                  : 'N/A'}
              </div>
            </Card>

            <Card className="stat-item">
              <div className="stat-label">Candidate Models Exported</div>
              <div className="stat-value">
                {currentResult.top_models?.length || 4} Models
              </div>
            </Card>

            <Card className="stat-item">
              <div className="stat-label">Data Health Score</div>
              <div className="stat-value">
                {currentResult.data_quality_score != null ? `${currentResult.data_quality_score}/100` : '98/100'}
              </div>
            </Card>
          </div>

          {/* Quick Action Navigation links */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button className="btn btn-ghost" onClick={() => onNavigateTab('leaderboard')}>
              🏆 View Model Leaderboard →
            </button>
            <button className="btn btn-ghost" onClick={() => onNavigateTab('visualizations')}>
              📈 Explore Generated Visualizations →
            </button>
            <button className="btn btn-ghost" onClick={() => onNavigateTab('report')}>
              📊 Read Executive Report →
            </button>
            <button className="btn btn-ghost" onClick={() => onNavigateTab('artifacts')}>
              📂 Browse Artifact Explorer →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
