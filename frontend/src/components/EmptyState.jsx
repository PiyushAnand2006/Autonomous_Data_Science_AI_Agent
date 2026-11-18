import React from 'react';
import { Icon } from './Icon';

export function EmptyState({ icon = 'inbox', title, desc, action }) {
  return (
    <div className="empty-state rise-in">
      <div className="empty-state-icon">
        <Icon name={icon} size={26} />
      </div>
      <div className="empty-state-title">{title}</div>
      <div className="empty-state-desc">{desc}</div>
      {action}
    </div>
  );
}
