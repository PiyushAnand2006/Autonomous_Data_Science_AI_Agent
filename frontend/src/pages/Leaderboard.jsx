import React from 'react';
import { ModelCard } from '../components/ModelCard';
import { Card } from '../components/Card';

export function Leaderboard({ result }) {
  if (!result) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🏆</div>
        <div className="empty-state-title">No Model Leaderboard Available</div>
        <div className="empty-state-desc">
          Launch an autonomous pipeline from the Pipeline tab to train and benchmark candidate machine learning algorithms.
        </div>
      </div>
    );
  }

  const topModels = result.top_models || [];
  const allCandidates = result.all_candidates || (topModels.length > 0 ? topModels.map((m, i) => ({
    rank: m.rank || i + 1,
    model: m.model_name,
    training_time: m.training_time,
    accuracy: m.metrics?.accuracy,
    f1: m.metrics?.f1,
    precision: m.metrics?.precision,
    recall: m.metrics?.recall,
    roc_auc: m.metrics?.roc_auc,
  })) : []);

  // Determine dynamic metric columns based on available keys
  const metricKeys = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'rmse', 'mae', 'r2', 'silhouette'].filter(
    (key) => allCandidates.some((c) => c[key] !== undefined && c[key] !== null)
  );

  return (
    <div>
      <div style={{ marginBottom: '28px' }}>
        <div className="eyebrow">Evaluated Models & Artifacts</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: '4px 0 8px 0' }}>
          🏆 Top 4 Selected Models (Exported to 06_Models/)
        </h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          The autonomous evaluation agent validated all candidates against isolated holdout partitions. The top 4 architectures were exported as production-ready serialized binaries.
        </p>
      </div>

      {/* Top Models Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '40px' }}>
        {topModels.map((model, idx) => (
          <ModelCard
            key={idx}
            model={{ ...model, rank: model.rank || idx + 1 }}
            runId={result.run_id}
          />
        ))}
      </div>

      {/* Full Benchmark of All Evaluated Candidates */}
      <div style={{ marginTop: '36px' }}>
        <h3 style={{ fontSize: '1.35rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>📋</span> Complete Benchmark of All Evaluated Candidates
        </h3>

        {allCandidates.length > 0 ? (
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <div className="data-table-wrap">
              <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--surface-3)', borderBottom: '1px solid var(--border-subtle)' }}>
                    <th style={{ padding: '12px 16px', textAlign: 'center', width: '70px', fontWeight: 600 }}>Rank</th>
                    <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600 }}>Model</th>
                    <th style={{ padding: '12px 16px', textAlign: 'right', fontWeight: 600 }}>Training Time (s)</th>
                    {metricKeys.map((k) => (
                      <th key={k} style={{ padding: '12px 16px', textAlign: 'right', fontWeight: 600, textTransform: 'uppercase', fontSize: 'var(--text-xs)' }}>
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {allCandidates.map((c, i) => {
                    const isBest = c.rank === 1 || i === 0;
                    return (
                      <tr
                        key={i}
                        style={{
                          background: isBest ? 'rgba(16, 185, 129, 0.15)' : (i % 2 === 0 ? 'var(--surface-1)' : 'transparent'),
                          borderBottom: '1px solid var(--border-subtle)',
                          fontWeight: isBest ? 700 : 400,
                        }}
                      >
                        <td style={{ padding: '12px 16px', textAlign: 'center', color: isBest ? '#10b981' : 'var(--text-secondary)' }}>
                          {c.rank || i + 1}
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'left', color: isBest ? '#10b981' : 'var(--text-primary)' }}>
                          <strong>{c.model}</strong>
                        </td>
                        <td style={{ padding: '12px 16px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: isBest ? '#10b981' : 'var(--text-secondary)' }}>
                          {Number(c.training_time || 0).toFixed(6)}
                        </td>
                        {metricKeys.map((k) => (
                          <td
                            key={k}
                            style={{
                              padding: '12px 16px',
                              textAlign: 'right',
                              fontFamily: 'var(--font-mono)',
                              fontSize: 'var(--text-sm)',
                              color: isBest ? '#10b981' : 'var(--text-primary)',
                            }}
                          >
                            {c[k] !== undefined && c[k] !== null ? Number(c[k]).toFixed(6) : '—'}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ padding: '12px 16px', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', background: 'var(--surface-2)', borderTop: '1px solid var(--border-subtle)' }}>
              Showing all {allCandidates.length} evaluated model candidates, ranked by performance metric (higher is better). The best model is highlighted in green.
            </div>
          </Card>
        ) : (
          <Card>
            <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
              No complete benchmark table data recorded for this run.
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
