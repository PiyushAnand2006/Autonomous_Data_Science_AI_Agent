import React, { useEffect, useRef } from 'react';
import { Icon } from './Icon';

/**
 * Parse a backend log line into { tag, text, tagClass }.
 * Every agent gets a slot in the warm Foundry tag spectrum —
 * amber, clay, rust, olive, jade, copper, sand, cocoa, ember.
 * No purple, no blue anywhere.
 */
export function parseLogLine(log) {
  if (!log) return { tag: null, text: '', tagClass: 'tag-ember' };

  // Strip raw unicode emojis the backend may emit
  let clean = log
    .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}📦🔍📋🛡️📊🧹✂️🔒⚙️🛠️🤖📈📓📄]/gu, '')
    .trim();

  const tagMatch = clean.match(/^[◈⬡✦⚡⚠❯\s]*\[(.*?)\]\s*(.*)$/);
  if (tagMatch) {
    const rawTag = tagMatch[1];
    const text = tagMatch[2];
    const upper = rawTag.toUpperCase();

    let tagClass = 'tag-ember';
    if (upper.includes('INIT')) tagClass = 'tag-cocoa';
    else if (upper.includes('PROFILER') || upper.includes('PROFILE')) tagClass = 'tag-copper';
    else if (upper.includes('PLAN')) tagClass = 'tag-sand';
    else if (upper.includes('QUALITY') || upper.includes('AUDIT')) tagClass = 'tag-jade';
    else if (upper.includes('EDA') || upper.includes('VISUAL')) tagClass = 'tag-olive';
    else if (upper.includes('CLEAN')) tagClass = 'tag-copper';
    else if (upper.includes('SPLIT')) tagClass = 'tag-sand';
    else if (upper.includes('LEAKAGE') || upper.includes('GUARD')) tagClass = 'tag-rust';
    else if (upper.includes('FEATURE')) tagClass = 'tag-clay';
    else if (upper.includes('PREPROCESS') || upper.includes('ENCODER')) tagClass = 'tag-copper';
    else if (upper.includes('ML') || upper.includes('EXPERIMENT') || upper.includes('MODEL')) tagClass = 'tag-ember';
    else if (upper.includes('EVAL') || upper.includes('DIAGNOSTIC')) tagClass = 'tag-jade';
    else if (upper.includes('NOTEBOOK')) tagClass = 'tag-sand';
    else if (upper.includes('REPORT') || upper.includes('SUMMARY')) tagClass = 'tag-clay';
    else if (upper.includes('WARN')) tagClass = 'tag-rust';

    return { tag: rawTag, text: text || clean, tagClass };
  }

  // Fallback: infer the agent from message content when the backend sends untagged text
  let inferredTag = null;
  let tagClass = 'tag-ember';
  const lower = clean.toLowerCase();

  if (lower.includes('initializ')) { inferredTag = 'INIT'; tagClass = 'tag-cocoa'; }
  else if (lower.includes('profil')) { inferredTag = 'PROFILER'; tagClass = 'tag-copper'; }
  else if (lower.includes('plan') || lower.includes('strateg')) { inferredTag = 'AI PLANNER'; tagClass = 'tag-sand'; }
  else if (lower.includes('quality') || lower.includes('audit') || lower.includes('anomal')) { inferredTag = 'DATA QUALITY'; tagClass = 'tag-jade'; }
  else if (lower.includes('exploratory') || lower.includes('chart') || lower.includes('correlation') || lower.includes('eda')) { inferredTag = 'EDA'; tagClass = 'tag-olive'; }
  else if (lower.includes('sanitiz') || lower.includes('clean') || lower.includes('deduplicat')) { inferredTag = 'CLEANER'; tagClass = 'tag-copper'; }
  else if (lower.includes('split') || lower.includes('train/test') || lower.includes('holdout')) { inferredTag = 'SPLITTER'; tagClass = 'tag-sand'; }
  else if (lower.includes('leakage') || lower.includes('guard')) { inferredTag = 'LEAKAGE GUARD'; tagClass = 'tag-rust'; }
  else if (lower.includes('feature')) { inferredTag = 'AI FEATURE ENG'; tagClass = 'tag-clay'; }
  else if (lower.includes('preprocess') || lower.includes('encoder') || lower.includes('scaling')) { inferredTag = 'PREPROCESS'; tagClass = 'tag-copper'; }
  else if (lower.includes('training') || lower.includes('leaderboard') || lower.includes('model') || lower.includes('experiment')) { inferredTag = 'ML EXPERIMENT'; tagClass = 'tag-ember'; }
  else if (lower.includes('evaluat') || lower.includes('residual') || lower.includes('metric')) { inferredTag = 'EVALUATION'; tagClass = 'tag-jade'; }
  else if (lower.includes('notebook') || lower.includes('jupyter')) { inferredTag = 'NOTEBOOK'; tagClass = 'tag-sand'; }
  else if (lower.includes('report') || lower.includes('summary') || lower.includes('executive')) { inferredTag = 'EXECUTIVE REPORT'; tagClass = 'tag-clay'; }

  return {
    tag: inferredTag,
    text: clean.replace(/^[◈⬡✦⚡⚠❯\s]+/, ''),
    tagClass,
  };
}

export function PipelineConsole({ logs, status, isRunning, runId, error }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [logs]);

  const isError = status === 'error' || Boolean(error);
  const isDone = status === 'completed' && !isRunning;

  return (
    <section className="console rise-in" aria-live="polite">
      <div className="console-head">
        <div className="console-title">
          <span className={`console-status-dot ${isRunning ? 'running' : isError ? 'error' : isDone ? 'done' : ''}`} />
          <span>
            {isRunning
              ? 'Autonomous pipeline running'
              : isError
              ? 'Pipeline error'
              : isDone
              ? 'Pipeline completed'
              : 'Pipeline console'}
          </span>
        </div>
        {runId && <span className="console-runid">RUN · {runId}</span>}
      </div>

      <div className="console-stream" role="log">
        {logs.map((log, idx) => {
          const isLatest = idx === logs.length - 1;
          const parsed = parseLogLine(log);
          return (
            <div key={idx} className={`log-line ${isLatest && isRunning ? 'latest' : ''}`}>
              {parsed.tag && <span className={`log-tag ${parsed.tagClass}`}>{parsed.tag}</span>}
              <span className="log-text">{parsed.text}</span>
              {isLatest && isRunning && <span className="cursor-blink" />}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {isError && error && (
        <div className="console-error">
          <strong style={{ color: '#f0c2ba' }}>Error details:</strong> {error}
        </div>
      )}
    </section>
  );
}
