import React from 'react';
import { StatusPill } from './StatusPill';

export function Navbar({ activeTab, setActiveTab, onOpenSettings, executionMode }) {
  const tabs = [
    { id: 'dashboard', label: '🚀 Launch Pipeline' },
    { id: 'report', label: '📊 In-Depth Executive Report' },
    { id: 'visualizations', label: '📈 Visualizations' },
    { id: 'leaderboard', label: '🏆 Top Models Leaderboard' },
    { id: 'artifacts', label: '📦 Dataset Versions & Artifacts' },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-brand" onClick={() => setActiveTab('dashboard')}>
        <div className="navbar-brand-icon">🧠</div>
        <span>AADS</span>
      </div>

      <div className="navbar-links">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`navbar-link ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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
  );
}
