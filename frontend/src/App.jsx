import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ThemeSwitch } from './components/ThemeSwitch';
import { SettingsDrawer } from './components/SettingsDrawer';
import { Dashboard } from './pages/Dashboard';
import { Leaderboard } from './pages/Leaderboard';
import { Visualizations } from './pages/Visualizations';
import { Report } from './pages/Report';
import { Artifacts } from './pages/Artifacts';
import { getSettings } from './api';

export default function App() {
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

  // Fresh in-memory session only — same policy as the original frontend.
  const [currentResult, setCurrentResult] = useState(null);

  useEffect(() => {
    try {
      localStorage.removeItem('aads_latest_result');
    } catch (e) {
      /* ignore */
    }
    loadInitialSettings();
  }, []);

  const loadInitialSettings = async () => {
    try {
      const res = await getSettings();
      if (res.settings) {
        setSettings((prev) => ({
          ...prev,
          ...res.settings,
          execution_mode: 'local', // Always default to Local mode on reload
        }));
      }
    } catch (e) {
      console.warn('Backend not ready yet, using defaults', e);
    }
  };

  const tabViews = {
    dashboard: (
      <Dashboard
        settings={settings}
        setSettings={setSettings}
        onOpenSettings={() => setIsSettingsOpen(true)}
        currentResult={currentResult}
        setCurrentResult={setCurrentResult}
        onNavigateTab={setActiveTab}
      />
    ),
    leaderboard: <Leaderboard result={currentResult} />,
    visualizations: <Visualizations result={currentResult} />,
    report: <Report result={currentResult} />,
    artifacts: <Artifacts result={currentResult} />,
  };

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenSettings={() => setIsSettingsOpen(true)}
        executionMode={settings.execution_mode}
      />

      <main className="app-main">
        <div className="content-col" key={activeTab}>
          {tabViews[activeTab]}
        </div>
      </main>

      <ThemeSwitch />

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
