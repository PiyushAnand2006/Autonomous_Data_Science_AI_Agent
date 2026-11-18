import React, { useState, useRef, useEffect } from 'react';
import { Icon } from './Icon';

/**
 * Find the nearest scrollable ancestor's rect so the menu can pick the
 * direction with real room (drawer bodies clip absolutely-positioned menus).
 */
function getClipRect(el) {
  let node = el ? el.parentElement : null;
  while (node && node !== document.body) {
    const st = getComputedStyle(node);
    const oy = st.overflowY;
    if ((oy === 'auto' || oy === 'scroll' || oy === 'overlay') && node.scrollHeight > node.clientHeight) {
      return node.getBoundingClientRect();
    }
    node = node.parentElement;
  }
  return { top: 0, bottom: window.innerHeight, left: 0, right: window.innerWidth };
}

const PREFERRED_MAX = 260;
const MIN_MENU = 120;
const SAFETY = 14;

/**
 * Searchable combo box for model identifiers.
 * - Opens with the FULL option list (e.g. the backend's provider model cache);
 *   filtering kicks in only once the user types.
 * - Flips up/down automatically based on the space inside the nearest
 *   scrollable ancestor, and clamps its height to the room available.
 * - Typed custom values are supported; keyboard: ↑↓ navigate, Enter select, Esc close.
 */
export function SearchableSelect({
  value = '',
  onChange,
  options = [],
  placeholder = 'Type to search or select a model…',
  direction = 'down',
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState(value || '');
  const [highlightIdx, setHighlightIdx] = useState(-1);
  const [showAll, setShowAll] = useState(true);
  const [menuStyle, setMenuStyle] = useState({ dir: direction, maxH: PREFERRED_MAX });
  const containerRef = useRef(null);
  const menuRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const v = value || '';
    // Sync from the prop, but skip the reset when it's just the echo of our
    // own typing (parent reflects onChange back into `value`) — otherwise the
    // filter would be undone on every keystroke and the menu never narrows.
    setQuery((prev) => {
      if (prev !== v) setShowAll(true);
      return v;
    });
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Scrolls outside the open menu (drawer body, page, window) close it —
  // they would misalign the anchored menu. Scrolling the model list itself
  // must NOT close it, so events originating inside the menu are ignored.
  useEffect(() => {
    if (!isOpen) return undefined;
    const onScroll = (e) => {
      if (menuRef.current && e.target instanceof Node && menuRef.current.contains(e.target)) return;
      setIsOpen(false);
    };
    const onResize = () => setIsOpen(false);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [isOpen]);

  const filteredOptions = showAll || !query
    ? options
    : options.filter((opt) => opt.toLowerCase().includes(query.toLowerCase()));

  const openMenu = () => {
    setShowAll(true); // browse the full backend list, not just the current match
    setHighlightIdx(-1);

    // Pick the direction with real room inside the clipping container
    let dir = direction;
    let maxH = PREFERRED_MAX;
    const input = inputRef.current;
    if (input) {
      const r = input.getBoundingClientRect();
      const clip = getClipRect(input);
      const spaceAbove = r.top - clip.top - SAFETY;
      const spaceBelow = clip.bottom - r.bottom - SAFETY;
      if (direction === 'up' && spaceAbove < MIN_MENU && spaceBelow > spaceAbove) dir = 'down';
      else if (direction === 'down' && spaceBelow < MIN_MENU && spaceAbove > spaceBelow) dir = 'up';
      maxH = Math.max(MIN_MENU, Math.min(PREFERRED_MAX, (dir === 'up' ? spaceAbove : spaceBelow)));
    }
    setMenuStyle({ dir, maxH });
    setIsOpen(true);
  };

  const handleSelect = (val) => {
    setQuery(val);
    setShowAll(true);
    if (onChange) onChange(val);
    setIsOpen(false);
    inputRef.current?.blur();
  };

  const handleInputChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    setShowAll(false); // typing filters
    if (onChange) onChange(val);
    setIsOpen(true);
    setHighlightIdx(-1);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!isOpen) openMenu();
      else setHighlightIdx((i) => Math.min(i + 1, filteredOptions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIdx((i) => Math.max(i - 1, -1));
    } else if (e.key === 'Enter') {
      if (isOpen && highlightIdx >= 0 && filteredOptions[highlightIdx]) {
        e.preventDefault();
        handleSelect(filteredOptions[highlightIdx]);
      } else if (isOpen) {
        setIsOpen(false);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <div className="combo" ref={containerRef}>
      <div className="combo-input-wrap">
        <input
          ref={inputRef}
          type="text"
          className="input input-mono"
          style={{ paddingRight: 40 }}
          value={query}
          onChange={handleInputChange}
          onFocus={openMenu}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          role="combobox"
          aria-expanded={isOpen}
          aria-autocomplete="list"
        />
        <button
          type="button"
          className="combo-toggle"
          onClick={() => (isOpen ? setIsOpen(false) : openMenu())}
          aria-label={isOpen ? 'Close model list' : 'Open model list'}
        >
          <Icon name="chevron-down" size={15} className="accordion-chevron" />
        </button>
      </div>

      {isOpen && (
        <div
          ref={menuRef}
          className={`combo-menu ${menuStyle.dir === 'up' ? 'up' : 'down'}`}
          style={{ maxHeight: menuStyle.maxH }}
          role="listbox"
        >
          {filteredOptions.length > 0 ? (
            filteredOptions.map((opt, idx) => {
              const isSelected = opt === value;
              return (
                <button
                  key={opt}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  className={`combo-option ${isSelected ? 'selected' : ''} ${idx === highlightIdx ? 'highlighted' : ''}`}
                  onClick={() => handleSelect(opt)}
                  onMouseEnter={() => setHighlightIdx(idx)}
                >
                  <span style={{ wordBreak: 'break-all' }}>{opt}</span>
                  {isSelected && <Icon name="check" size={14} />}
                </button>
              );
            })
          ) : (
            <div className="combo-empty">
              No listed match — custom model: <strong>“{query}”</strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
