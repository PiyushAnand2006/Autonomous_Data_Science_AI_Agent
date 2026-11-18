import React from 'react';
import { ModelCard } from '../components/ModelCard';
import { Card } from '../components/Card';
import { EmptyState } from '../components/EmptyState';
import { Icon } from '../components/Icon';

export function Leaderboard({ result }) {
  if (!result) {
    return (
      <>
        <PageHead />
        <EmptyState
          icon="trophy"
          title="No leaderboard yet"
          desc="Launch an autonomous pipeline from the Launch Pipeline tab to train and benchmark candidate machine-learning algorithms."
        />
      </>
    );
  }

  const topModels = result.top_models || [];
  const rawCandidates = result.all_candidates || (topModels.length > 0
    ? topModels.map((m, i) => ({
        rank: m.rank || i + 1,
        model: m.model_name,
        training_time: m.training_time,
        accuracy: m.metrics?.accuracy,
        f1: m.metrics?.f1,
        precision: m.metrics?.precision,
        recall: m.metrics?.recall,
        roc_auc: m.metrics?.roc_auc,
      }))
    : []);

  // Rank by the first available metric: F1 > Accuracy > ROC-AUC > RMSE (asc) > R² > Silhouette
  const allCandidates = [...rawCandidates]
    .sort((a, b) => {
      if (a.f1 != null && b.f1 != null && Math.abs(b.f1 - a.f1) > 1e-6) return b.f1 - a.f1;
      if (a.accuracy != null && b.accuracy != null && Math.abs(b.accuracy - a.accuracy) > 1e-6) return b.accuracy - a.accuracy;
      if (a.roc_auc != null && b.roc_auc != null && Math.abs(b.roc_auc - a.roc_auc) > 1e-6) return b.roc_auc - a.roc_auc;
      if (a.rmse != null && b.rmse != null && Math.abs(a.rmse - b.rmse) > 1e-6) return a.rmse - b.rmse;
      if (a.r2 != null && b.r2 != null && Math.abs(b.r2 - a.r2) > 1e-6) return b.r2 - a.r2;
      if (a.silhouette != null && b.silhouette != null && Math.abs(b.silhouette - a.silhouette) > 1e-6) return b.silhouette - a.silhouette;
      return (a.rank || 0) - (b.rank || 0);
    })
    .map((cand, idx) => ({ ...cand, rank: idx + 1 }));

  const metricKeys = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'rmse', 'mae', 'r2', 'silhouette'].filter(
    (key) => allCandidates.some((c) => c[key] !== undefined && c[key] !== null)
  );

  return (
    <>
      <PageHead />

      {/* Top exported models */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {topModels.map((model, idx) => (
          <ModelCard key={idx} model={{ ...model, rank: model.rank || idx + 1 }} runId={result.run_id} />
        ))}
      </div>

      {/* Full benchmark */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <h3 className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Icon name="grid" size={18} style={{ color: 'var(--ember-1)' }} />
          Complete benchmark — all evaluated candidates
        </h3>

        {allCandidates.length > 0 ? (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 70, textAlign: 'center' }}>Rank</th>
                  <th>Model</th>
                  <th style={{ textAlign: 'right' }}>Time (s)</th>
                  {metricKeys.map((k) => (
                    <th key={k} style={{ textAlign: 'right' }}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allCandidates.map((c, i) => {
                  const isBest = c.rank === 1 || i === 0;
                  return (
                    <tr key={i} className={isBest ? 'row-best' : ''}>
                      <td style={{ textAlign: 'center', fontWeight: isBest ? 700 : 400 }}>{c.rank || i + 1}</td>
                      <td>
                        <strong style={{ color: isBest ? 'var(--jade)' : 'var(--ink)', fontWeight: isBest ? 700 : 600 }}>
                          {c.model}
                        </strong>
                      </td>
                      <td className="num">{Number(c.training_time || 0).toFixed(6)}</td>
                      {metricKeys.map((k) => (
                        <td key={k} className="num">
                          {c[k] !== undefined && c[k] !== null ? Number(c[k]).toFixed(6) : '—'}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="table-footnote">
              All {allCandidates.length} evaluated candidates, ranked by performance metric (higher is better —
              RMSE lower is better). The champion row is highlighted in jade.
            </div>
          </div>
        ) : (
          <Card>
            <div style={{ textAlign: 'center', padding: 24, color: 'var(--ink-3)' }}>
              No complete benchmark table was recorded for this run.
            </div>
          </Card>
        )}
      </section>
    </>
  );
}

function PageHead() {
  return (
    <header className="page-head">
      <div className="eyebrow">Evaluated Models &amp; Artifacts</div>
      <h1 className="page-title">
        Model <span className="gradient-ink">leaderboard</span>
      </h1>
      <p className="page-lede">
        The top {4} architectures were validated on isolated holdout partitions and exported as
        production-ready serialized binaries to <code>06_Models/</code>.
      </p>
      <hr className="head-rule" />
    </header>
  );
}
