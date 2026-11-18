import React, { useEffect, useState } from 'react';
import { Icon, BrandMark } from './Icon';
import { checkHealth } from '../api';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Launch Pipeline', icon: 'play' },
  { id: 'leaderboard', label: 'Leaderboard', icon: 'trophy' },
  { id: 'visualizations', label: 'Visualizations', icon: 'chart' },
  { id: 'report', label: 'Executive Report', icon: 'report' },
  { id: 'artifacts', label: 'Artifacts', icon: 'package' },
];

function readCollapsed() {
  try {
    return localStorage.getItem('audas_sidebar') === 'collapsed';
  } catch (e) {
    return false;
  }
}

export function Sidebar({ activeTab, setActiveTab, onOpenSettings, executionMode }) {
  const [backend, setBackend] = useState('checking');
  const [collapsed, setCollapsed] = useState(readCollapsed);

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('audas_sidebar', next ? 'collapsed' : 'expanded');
      } catch (e) {
        /* ignore */
      }
      return next;
    });
  };

  useEffect(() => {
    let cancelled = false;
    const probe = async () => {
      try {
        await checkHealth();
        if (!cancelled) setBackend('online');
      } catch (e) {
        if (!cancelled) setBackend('offline');
      }
    };
    probe();
    const timer = setInterval(probe, 30000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const isAi = executionMode === 'ai';

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-head">
        <button
          className="sidebar-brand"
          onClick={() => setActiveTab('dashboard')}
          aria-label="AUDAS home"
          title="Go to pipeline launcher"
        >
          <BrandMark size={40} />
          <span className="brand-text">
            <span className="brand-name">AUDAS</span>
            <span className="brand-sub">Autonomous Data Scientist</span>
          </span>
          <span className="sr-only">Go to pipeline launcher</span>
        </button>

        <button
          type="button"
          className="sidebar-collapse"
          onClick={toggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <Icon name={collapsed ? 'panel-expand' : 'panel-collapse'} size={16} />
        </button>
      </div>

      <nav className="sidebar-nav" aria-label="Primary">
        <div className="sidebar-section-label">Workspace</div>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
            onClick={() => setActiveTab(item.id)}
            aria-current={activeTab === item.id ? 'page' : undefined}
            title={item.label}
          >
            <Icon name={item.icon} size={17} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-foot">
        <div className="backend-status" title={`Backend ${backend}`}>
          <span className={`status-dot ${backend === 'online' ? 'online' : backend === 'offline' ? 'offline' : ''}`} />
          <span>
            Backend {backend}
          </span>
        </div>

        <button className="nav-item" onClick={onOpenSettings} aria-haspopup="dialog" title="Model Settings">
          <Icon name="settings" size={17} />
          <span>Model Settings</span>
          {isAi && <span className="chip chip-ember" style={{ marginLeft: 'auto' }}>AI</span>}
        </button>
      </div>
    </aside>
  );
}
