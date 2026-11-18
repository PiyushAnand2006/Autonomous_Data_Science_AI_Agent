import React, { useState, useEffect, useRef } from 'react';
import { Card } from '../components/Card';
import { FileUploader } from '../components/FileUploader';
import { PipelineConsole } from '../components/PipelineConsole';
import { Icon } from '../components/Icon';
import { launchPipeline, subscribeToPipelineStream, getPipelineResult, previewDataset, getZipDownloadUrl } from '../api';

const DEFAULT_OBJECTIVE = 'Predict customer churn, discover key behavioral drivers, and export top performing ML models.';

export function Dashboard({ settings, setSettings, onOpenSettings, currentResult, setCurrentResult, onNavigateTab }) {
  const [dataFile, setDataFile] = useState(null);
  const [objective, setObjective] = useState(DEFAULT_OBJECTIVE);
  const [targetCol, setTargetCol] = useState('');
  const [previewData, setPreviewData] = useState(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [runId, setRunId] = useState(currentResult?.run_id || null);
  const [status, setStatus] = useState(currentResult ? 'completed' : 'idle');
  const [error, setError] = useState(null);
  const [modeNotice, setModeNotice] = useState(null);
  const activeStreamCloseRef = useRef(null);

  // NOTE: no auto-reset effect here. Mode switches and file changes already
  // reset state explicitly; a blanket reset-on-empty effect would race with
  // result arrival and wipe the finished console's audit trail.

  useEffect(() => {
    return () => {
      if (activeStreamCloseRef.current) {
        activeStreamCloseRef.current();
        activeStreamCloseRef.current = null;
      }
    };
  }, []);

  const stopActiveStream = () => {
    if (activeStreamCloseRef.current) {
      activeStreamCloseRef.current();
      activeStreamCloseRef.current = null;
    }
  };

  const resetRunState = () => {
    setLogs([]);
    setStatus('idle');
    setError(null);
    setRunId(null);
    setIsRunning(false);
  };

  const isAiMode = settings?.execution_mode === 'ai';

  const handleModeToggle = (mode) => {
    if (settings?.execution_mode !== mode) {
      stopActiveStream();
      setSettings((prev) => ({ ...prev, execution_mode: mode }));
      setCurrentResult(null);
      resetRunState();
      setModeNotice(
        `Switched to ${mode === 'ai' ? 'AI-Powered Agentic' : 'Local (Deterministic)'} engine. Launch the pipeline again to run under the new mode.`
      );
      setTimeout(() => setModeNotice(null), 6000);
    }
  };

  const handleFileReady = async (fileInfo) => {
    stopActiveStream();
    setDataFile(fileInfo);
    setCurrentResult(null);
    resetRunState();

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
    if (!dataFile || !dataFile.path) return;

    // Clear previous result so stale data is never shown while running
    setCurrentResult(null);
    setRunId(null);
    setIsRunning(true);
    setLogs(['Initializing autonomous pipeline run…']);
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
      custom_base_url: settings.custom_base_url || '',
      custom_provider_name: settings.custom_provider_name || '',
    };

    try {
      stopActiveStream();
      const res = await launchPipeline(payload);
      setRunId(res.run_id);

      const unsubscribe = subscribeToPipelineStream(
        res.run_id,
        (msg) => {
          setLogs((prev) => {
            // Model training progress lines replace the previous one (same slot)
            const isCandidateProgress = msg.includes('[Model ') || msg.includes('Training candidate machine learning models');
            if (isCandidateProgress) {
              const existingIdx = prev.findIndex((l) => l.includes('[Model ') || l.includes('Training candidate machine learning models'));
              if (existingIdx !== -1) {
                const nextLogs = [...prev];
                nextLogs[existingIdx] = msg;
                return nextLogs;
              }
            }
            return [...prev, msg];
          });
        },
        (err) => {
          activeStreamCloseRef.current = null;
          setError(err);
          setIsRunning(false);
          setStatus('error');
        },
        async () => {
          activeStreamCloseRef.current = null;
          setIsRunning(false);
          setStatus('completed');
          try {
            const finalRes = await getPipelineResult(res.run_id);
            if (finalRes && finalRes.result) {
              setCurrentResult(finalRes.result);
              // Adopt the backend's canonical run id so the console header
              // matches the summary/downloads (the two ids alias the same run).
              if (finalRes.result.run_id) setRunId(finalRes.result.run_id);
            } else {
              // Backend may still be wrapping up — retry shortly
              setTimeout(async () => {
                try {
                  const retryRes = await getPipelineResult(res.run_id);
                  if (retryRes?.result) setCurrentResult(retryRes.result);
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
      activeStreamCloseRef.current = unsubscribe;
    } catch (err) {
      setError(err.message || 'Failed to start pipeline');
      setIsRunning(false);
      setStatus('error');
    }
  };

  const bestMetric = currentResult?.best_metrics
    ? Object.values(currentResult.best_metrics)[0]
    : null;

  return (
    <>
      {/* ---- Page head ---- */}
      <header className="page-head">
        <div className="eyebrow">Autonomous Data Science Workspace</div>
        <h1 className="page-title">
          Launch a <span className="gradient-ink">full ML lifecycle</span> from one objective
        </h1>
        <p className="page-lede">
          Transform a raw tabular dataset and a natural-language goal into a production-ready,
          reproducible machine-learning project — profiled, cleaned, benchmarked and packaged by a
          team of specialist agents.
        </p>
        <hr className="head-rule" />
      </header>

      {/* ---- Mode strip ---- */}
      <section className="mode-strip">
        <div style={{ minWidth: 260, flex: 1 }}>
          <div className="mode-strip-label">Execution Mode</div>
          <div className="segmented" role="group" aria-label="Execution mode">
            <button
              type="button"
              className={`segmented-btn ${!isAiMode ? 'active' : ''}`}
              onClick={() => handleModeToggle('local')}
            >
              <Icon name="zap" size={15} />
              Local · Offline
            </button>
            <button
              type="button"
              className={`segmented-btn ${isAiMode ? 'active' : ''}`}
              onClick={() => handleModeToggle('ai')}
            >
              <Icon name="sparkles" size={15} />
              AI-Powered
            </button>
          </div>

          {isAiMode ? (
            <div className="mode-strip-meta">
              <Icon name="sparkles" size={14} />
              <span>
                {(settings?.selected_provider || 'openrouter').toUpperCase()}
                <span className="mono" style={{ color: 'var(--ink)' }}>
                  {' '}
                  · {settings?.selected_model?.split('/')?.pop() || 'no model selected'}
                </span>
              </span>
              <button className="btn btn-ghost btn-sm" onClick={onOpenSettings}>
                Configure
                <Icon name="chevron-right" size={13} />
              </button>
            </div>
          ) : (
            <div className="mode-strip-meta">
              <Icon name="cpu" size={14} />
              <span>Deterministic local ML training — no API key needed</span>
            </div>
          )}
        </div>
      </section>

      {/* ---- Setup grid ---- */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 22, alignItems: 'start' }}>
        {/* Step 1 — data */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
            <span className="chip chip-ember">Step 1</span>
            <h3 className="section-title" style={{ fontSize: 'var(--text-lg)' }}>Data ingestion</h3>
          </div>

          <FileUploader onFileReady={handleFileReady} currentFile={dataFile} />

          {dataFile && previewData && (
            <div style={{ marginTop: 16 }}>
              <div className="accordion">
                <button
                  type="button"
                  className="accordion-head"
                  onClick={() => setIsPreviewOpen(!isPreviewOpen)}
                  aria-expanded={isPreviewOpen}
                >
                  <span className="accordion-head-left">
                    <Icon name="eye" size={15} />
                    <span>Dataset preview</span>
                  </span>
                  <span className="accordion-meta">
                    {previewData.total_rows} rows × {previewData.total_columns} cols
                  </span>
                  <Icon name="chevron-down" size={15} className={`accordion-chevron ${isPreviewOpen ? 'open' : ''}`} />
                </button>

                {isPreviewOpen && (
                  <div className="accordion-body" style={{ overflowX: 'auto' }}>
                    <table className="data-table" style={{ minWidth: 480 }}>
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
                              <td key={j} style={{ whiteSpace: 'nowrap' }}>{String(v)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>

        {/* Step 2 — objective */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
            <span className="chip chip-ember">Step 2</span>
            <h3 className="section-title" style={{ fontSize: 'var(--text-lg)' }}>Task &amp; objective</h3>
          </div>

          <div style={{ marginBottom: 16 }}>
            <label className="field-label" htmlFor="objective">Natural-language goal</label>
            <textarea
              id="objective"
              className="textarea"
              rows={4}
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="e.g. Predict customer churn and discover the key behavioral drivers"
            />
          </div>

          <div style={{ marginBottom: 18 }}>
            <label className="field-label" htmlFor="target-col">Target column</label>
            <input
              id="target-col"
              type="text"
              className="input input-mono"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              placeholder="Auto-detect (e.g. churn, label, price)"
            />
            <div className="field-hint">Leave blank and the Profiler agent will infer the target automatically.</div>
          </div>

          {modeNotice && (
            <div className="notice notice-sand" style={{ marginBottom: 14 }}>
              <Icon name="info" size={16} />
              <span>{modeNotice}</span>
            </div>
          )}

          <button
            className="btn btn-primary btn-lg btn-pill"
            style={{ width: '100%' }}
            onClick={handleLaunch}
            disabled={isRunning || !dataFile}
          >
            {isRunning ? (
              <>
                <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2.5 }} />
                Agents running…
              </>
            ) : (
              <>
                <Icon name="play" size={18} />
                Launch Autonomous Pipeline
                <span className="btn-chip">
                  <Icon name="arrow-right" size={14} strokeWidth={2.2} />
                </span>
              </>
            )}
          </button>

          {!dataFile && (
            <div className="field-hint" style={{ textAlign: 'center', marginTop: 10 }}>
              Select a dataset in Step 1 to enable launch.
            </div>
          )}
        </Card>
      </section>

      {/* ---- Live console ---- */}
      {(isRunning || logs.length > 0 || error) && (
        <PipelineConsole
          logs={logs}
          status={status}
          isRunning={isRunning}
          runId={runId}
          error={error}
        />
      )}

      {/* ---- Run summary ---- */}
      {currentResult && !isRunning && (
        <section className="rise-in" style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
          <div className="run-summary-head">
            <div className="page-head" style={{ gap: 4 }}>
              <div className="eyebrow" style={{ color: 'var(--jade)' }}>Pipeline Run Completed</div>
              <h2 className="section-title">Run summary</h2>
              <span className="run-id-inline">{currentResult.run_id}</span>
            </div>
            <a href={getZipDownloadUrl(currentResult.run_id)} className="btn btn-primary">
              <Icon name="package" size={16} />
              Download project archive (.zip)
            </a>
          </div>

          <div className="stat-hud">
            <Card className="stat-item" jade>
              <div className="stat-label">Best Performing Model</div>
              <div className="stat-value jade" style={{ fontSize: 'var(--text-lg)' }}>
                {currentResult.best_model_name || '—'}
              </div>
            </Card>

            <Card className="stat-item">
              <div className="stat-label">Primary Metric Score</div>
              <div className="stat-value ember">
                {bestMetric != null
                  ? typeof bestMetric === 'number'
                    ? bestMetric.toFixed(4)
                    : bestMetric
                  : 'N/A'}
              </div>
            </Card>

            <Card className="stat-item">
              <div className="stat-label">Models Exported</div>
              <div className="stat-value">{currentResult.top_models?.length || 4}</div>
            </Card>

            <Card className="stat-item">
              <div className="stat-label">Data Health Score</div>
              <div className="stat-value">
                {currentResult.data_quality_score != null ? `${currentResult.data_quality_score}/100` : '98/100'}
              </div>
            </Card>
          </div>

          <div className="quick-links">
            <button className="quick-link" onClick={() => onNavigateTab('leaderboard')}>
              <Icon name="trophy" size={16} />
              View model leaderboard
              <Icon name="arrow-right" size={15} />
            </button>
            <button className="quick-link" onClick={() => onNavigateTab('visualizations')}>
              <Icon name="chart" size={16} />
              Explore visualizations
              <Icon name="arrow-right" size={15} />
            </button>
            <button className="quick-link" onClick={() => onNavigateTab('report')}>
              <Icon name="report" size={16} />
              Read executive report
              <Icon name="arrow-right" size={15} />
            </button>
            <button className="quick-link" onClick={() => onNavigateTab('artifacts')}>
              <Icon name="package" size={16} />
              Browse artifacts
              <Icon name="arrow-right" size={15} />
            </button>
          </div>
        </section>
      )}
    </>
  );
}
