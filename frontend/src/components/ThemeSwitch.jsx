import React, { useState } from 'react';
import { Icon } from './Icon';

/**
 * Pill-style light/dark switch (top-right of the shell).
 * The knob slides between sun and moon; the active icon rides inside the knob
 * with ink that contrasts its gradient, the inactive one rests on the pill.
 */
export function ThemeSwitch() {
  const [isDark, setIsDark] = useState(() => document.documentElement.dataset.theme === 'dark');

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.dataset.theme = next ? 'dark' : 'light';
    try {
      localStorage.setItem('audas_theme', next ? 'dark' : 'light');
    } catch (e) {
      /* ignore */
    }
  };

  return (
    <div className="theme-switch-wrap">
      <span className="theme-switch-label" aria-hidden="true">
        {isDark ? 'Light' : 'Dark'}
      </span>
      <button
        type="button"
        className="theme-switch"
        role="switch"
        aria-checked={isDark}
        aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        onClick={toggle}
      >
        <span className="theme-switch-knob" />
        <Icon name="sun" size={16} strokeWidth={2} className="icon-sun" />
        <Icon name="moon" size={15} strokeWidth={2} className="icon-moon" />
      </button>
    </div>
  );
}
