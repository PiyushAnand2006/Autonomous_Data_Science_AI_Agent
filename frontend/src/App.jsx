import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { SettingsDrawer } from './components/SettingsDrawer';
import { Dashboard } from './pages/Dashboard';
import { Leaderboard } from './pages/Leaderboard';
import { Visualizations } from './pages/Visualizations';
import { Report } from './pages/Report';
import { Artifacts } from './pages/Artifacts';
import { getSettings } from './api';

export function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    execution_mode: 'local',
    selected_provider: 'openrouter',
    selected_model: '',
    autonomy: 'fully_autonomous',
    engine: 'pandas',
    random_seed: 42,
    api_keys: {},
  });

  // Fresh in-memory session only (no localStorage persistence across page reloads)
  const [currentResult, setCurrentResult] = useState(null);

  // Clear any existing localStorage cache on mount
  useEffect(() => {
    try {
      localStorage.removeItem('aads_latest_result');
    } catch (e) {
      // Ignore
    }
    loadInitialSettings();
  }, []);

  const loadInitialSettings = async () => {
    try {
      const res = await getSettings();
      if (res.settings) {
        setSettings((prev) => ({ ...prev, ...res.settings }));
      }
    } catch (e) {
      console.warn('Backend not ready yet, using defaults', e);
    }
  };

  return (
    <div className="app-layout">
      {/* Floating Glass Pill Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenSettings={() => setIsSettingsOpen(true)}
        executionMode={settings.execution_mode}
      />

      {/* Main Content Area — Kept mounted in DOM so state, running SSE, and tabs are never lost */}
      <main className="app-main">
        <div style={{ display: activeTab === 'dashboard' ? 'block' : 'none' }}>
          <Dashboard
            settings={settings}
            setSettings={setSettings}
            onOpenSettings={() => setIsSettingsOpen(true)}
            currentResult={currentResult}
            setCurrentResult={setCurrentResult}
            onNavigateTab={setActiveTab}
          />
        </div>

        <div style={{ display: activeTab === 'leaderboard' ? 'block' : 'none' }}>
          <Leaderboard result={currentResult} />
        </div>

        <div style={{ display: activeTab === 'visualizations' ? 'block' : 'none' }}>
          <Visualizations result={currentResult} />
        </div>

        <div style={{ display: activeTab === 'report' ? 'block' : 'none' }}>
          <Report result={currentResult} />
        </div>

        <div style={{ display: activeTab === 'artifacts' ? 'block' : 'none' }}>
          <Artifacts result={currentResult} />
        </div>
      </main>

      <SettingsDrawer
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSettingsSaved={(newSettings) => {
          setSettings((prev) => {
            if (newSettings.execution_mode && newSettings.execution_mode !== prev.execution_mode) {
              setCurrentResult(null);
            }
            return { ...prev, ...newSettings };
          });
        }}
      />
    </div>
  );
}

export default App;
