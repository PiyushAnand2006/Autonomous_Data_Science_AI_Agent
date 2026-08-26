import React from 'react';
import { StatusPill } from './StatusPill';

export function Navbar({ activeTab, setActiveTab, onOpenSettings, executionMode }) {
  const tabs = [
    { id: 'dashboard', label: '🚀 Launch Pipeline', shortLabel: '🚀 Launch' },
    { id: 'report', label: '📊 In-Depth Executive Report', shortLabel: '📊 Report' },
    { id: 'visualizations', label: '📈 Visualizations', shortLabel: '📈 Visuals' },
    { id: 'leaderboard', label: '🏆 Top Models Leaderboard', shortLabel: '🏆 Models' },
    { id: 'artifacts', label: '📦 Dataset Versions & Artifacts', shortLabel: '📦 Artifacts' },
  ];

  return (
    <header className="navbar-container">
      <nav className="navbar">
        <div className="navbar-brand" onClick={() => setActiveTab('dashboard')}>
          <div className="navbar-brand-icon">🧠</div>
          <span>AUDAS</span>
        </div>

        <div className="navbar-links">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`navbar-link ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-label-full">{tab.label}</span>
              <span className="tab-label-short">{tab.shortLabel}</span>
            </button>
          ))}
        </div>

        <div className="navbar-actions" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <StatusPill mode={executionMode} />
          <button
            className="navbar-settings"
            onClick={onOpenSettings}
            title="Open Settings & LLM Config"
          >
            ⚙️
          </button>
        </div>
      </nav>
    </header>
  );
}
