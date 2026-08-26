import React, { useEffect, useRef } from 'react';
import { Card } from './Card';

function parseLogLine(log) {
  if (!log) return { tag: null, text: '', tagClass: 'tag-purple', glyph: '◈' };

  // Remove any raw Unicode emojis or stray leading symbols
  let clean = log
    .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}📦🔍📋🛡️📊🧹✂️🔒⚙️🛠️🤖📈📓📄]/gu, '')
    .trim();

  // Match bracketed agent tag e.g. [PROFILER], [ML EXPERIMENT], etc.
  const tagMatch = clean.match(/^[◈⬡✦⚡⚠❯\s]*\[(.*?)\]\s*(.*)$/);
  if (tagMatch) {
    const rawTag = tagMatch[1];
    const text = tagMatch[2];

    let tagClass = 'tag-purple';
    let glyph = '◈';

    const upperTag = rawTag.toUpperCase();
    if (upperTag.includes('INIT')) {
      tagClass = 'tag-purple';
      glyph = '◈';
    } else if (upperTag.includes('PROFILER') || upperTag.includes('PROFILE')) {
      tagClass = 'tag-cyan';
      glyph = '⬡';
    } else if (upperTag.includes('PLAN')) {
      tagClass = 'tag-indigo';
      glyph = '✦';
    } else if (upperTag.includes('QUALITY') || upperTag.includes('AUDIT')) {
      tagClass = 'tag-emerald';
      glyph = '⬡';
    } else if (upperTag.includes('EDA') || upperTag.includes('VISUAL')) {
      tagClass = 'tag-amber';
      glyph = '◈';
    } else if (upperTag.includes('CLEAN')) {
      tagClass = 'tag-cyan';
      glyph = '⬡';
    } else if (upperTag.includes('SPLIT')) {
      tagClass = 'tag-indigo';
      glyph = '◈';
    } else if (upperTag.includes('LEAKAGE') || upperTag.includes('GUARD')) {
      tagClass = 'tag-rose';
      glyph = '⬡';
    } else if (upperTag.includes('FEATURE')) {
      tagClass = 'tag-purple';
      glyph = '✦';
    } else if (upperTag.includes('PREPROCESS') || upperTag.includes('ENCODER')) {
      tagClass = 'tag-cyan';
      glyph = '◈';
    } else if (upperTag.includes('ML') || upperTag.includes('EXPERIMENT') || upperTag.includes('MODEL')) {
      tagClass = 'tag-sapphire';
      glyph = '⚡';
    } else if (upperTag.includes('EVAL') || upperTag.includes('DIAGNOSTIC')) {
      tagClass = 'tag-emerald';
      glyph = '◈';
    } else if (upperTag.includes('NOTEBOOK')) {
      tagClass = 'tag-indigo';
      glyph = '⬡';
    } else if (upperTag.includes('REPORT') || upperTag.includes('SUMMARY')) {
      tagClass = 'tag-amber';
      glyph = '✦';
    } else if (upperTag.includes('WARN')) {
      tagClass = 'tag-rose';
      glyph = '⚠';
    }

    return { tag: rawTag, text: text || clean, tagClass, glyph };
  }

  // Fallback: Infer tag from log message content if backend sent untagged text
  let inferredTag = null;
  let tagClass = 'tag-purple';
  let glyph = '◈';
  const lower = clean.toLowerCase();

  if (lower.includes('initializ')) {
    inferredTag = 'INIT';
    tagClass = 'tag-purple';
    glyph = '◈';
  } else if (lower.includes('profil')) {
    inferredTag = 'PROFILER';
    tagClass = 'tag-cyan';
    glyph = '⬡';
  } else if (lower.includes('plan') || lower.includes('strateg')) {
    inferredTag = 'AI PLANNER';
    tagClass = 'tag-indigo';
    glyph = '✦';
  } else if (lower.includes('quality') || lower.includes('audit') || lower.includes('anomal')) {
    inferredTag = 'DATA QUALITY';
    tagClass = 'tag-emerald';
    glyph = '⬡';
  } else if (lower.includes('exploratory') || lower.includes('chart') || lower.includes('correlation') || lower.includes('eda')) {
    inferredTag = 'EDA';
    tagClass = 'tag-amber';
    glyph = '◈';
  } else if (lower.includes('sanitiz') || lower.includes('clean') || lower.includes('deduplicat')) {
    inferredTag = 'CLEANER';
    tagClass = 'tag-cyan';
    glyph = '⬡';
  } else if (lower.includes('split') || lower.includes('train/test') || lower.includes('holdout')) {
    inferredTag = 'SPLITTER';
    tagClass = 'tag-indigo';
    glyph = '◈';
  } else if (lower.includes('leakage') || lower.includes('guard')) {
    inferredTag = 'LEAKAGE GUARD';
    tagClass = 'tag-rose';
    glyph = '⬡';
  } else if (lower.includes('feature')) {
    inferredTag = 'AI FEATURE ENG';
    tagClass = 'tag-purple';
    glyph = '✦';
  } else if (lower.includes('preprocess') || lower.includes('encoder') || lower.includes('scaling')) {
    inferredTag = 'PREPROCESS';
    tagClass = 'tag-cyan';
    glyph = '◈';
  } else if (lower.includes('training') || lower.includes('leaderboard') || lower.includes('model') || lower.includes('experiment')) {
    inferredTag = 'ML EXPERIMENT';
    tagClass = 'tag-sapphire';
    glyph = '⚡';
  } else if (lower.includes('evaluat') || lower.includes('residual') || lower.includes('metric')) {
    inferredTag = 'EVALUATION';
    tagClass = 'tag-emerald';
    glyph = '◈';
  } else if (lower.includes('notebook') || lower.includes('jupyter')) {
    inferredTag = 'NOTEBOOK';
    tagClass = 'tag-indigo';
    glyph = '⬡';
  } else if (lower.includes('report') || lower.includes('summary') || lower.includes('executive')) {
    inferredTag = 'EXECUTIVE REPORT';
    tagClass = 'tag-amber';
    glyph = '✦';
  }

  return {
    tag: inferredTag,
    text: clean.replace(/^[◈⬡✦⚡⚠❯\s]+/, ''),
    tagClass,
    glyph,
  };
}

export function PipelineProgress({ logs, status, isRunning, runId, error }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const isError = status === 'error' || Boolean(error);

  return (
    <Card glow={isRunning ? 'sapphire' : isError ? '' : status === 'completed' ? 'emerald' : ''} style={{ marginTop: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            className={`phase-dot ${isRunning ? 'active' : isError ? '' : status === 'completed' ? 'done' : ''}`}
            style={isError ? { background: 'var(--accent-rose)', boxShadow: '0 0 10px rgba(244, 63, 94, 0.5)' } : {}}
          />
          <h4 style={{ margin: 0, color: isError ? 'var(--accent-rose)' : 'var(--text-primary)' }}>
            {isRunning
              ? 'Autonomous Multi-Agent Pipeline Running...'
              : isError
              ? 'Pipeline Error'
              : status === 'completed'
              ? 'Pipeline Completed Successfully'
              : 'Pipeline Execution'}
          </h4>
        </div>
        {runId && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            RUN ID: {runId}
          </span>
        )}
      </div>

      <div
        className="pipeline-progress"
        style={{
          maxHeight: '280px',
          overflowY: 'auto',
          background: 'var(--surface-1)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        {logs.map((log, idx) => {
          const isLatest = idx === logs.length - 1;
          const parsed = parseLogLine(log);

          return (
            <div key={idx} className={`pipeline-phase ${isLatest && isRunning ? 'active-phase' : ''}`} style={{ alignItems: 'center' }}>
              <div
                className={`phase-dot ${isLatest && isRunning ? 'active' : isLatest && isError ? '' : 'done'}`}
                style={isLatest && isError ? { background: 'var(--accent-rose)' } : {}}
              />
              <div
                className={`phase-text ${isLatest && isRunning ? 'active' : ''}`}
                style={{
                  fontFamily: 'var(--font-mono)',
                  color: isLatest && isError ? 'var(--accent-rose)' : undefined,
                  display: 'flex',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '6px',
                }}
              >
                {parsed.tag && (
                  <span className={`pipeline-tag ${parsed.tagClass}`}>
                    <span>{parsed.glyph}</span>
                    <span>{parsed.tag}</span>
                  </span>
                )}
                <span>{parsed.text}</span>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {isError && error && (
        <div
          style={{
            marginTop: '12px',
            padding: '12px 16px',
            background: 'rgba(244, 63, 94, 0.08)',
            border: '1px solid rgba(244, 63, 94, 0.25)',
            borderRadius: 'var(--radius-md)',
            color: '#fca5a5',
            fontSize: 'var(--text-sm)',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <strong>Error Details:</strong> {error}
        </div>
      )}
    </Card>
  );
}
