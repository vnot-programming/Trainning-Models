/**
 * Circadian Sync Utility — Bio-Digital Minimalism 2026
 * 
 * Sets data-time="morning|afternoon|evening|night" on <html> based on local time.
 * Enables CSS circadian color palettes for biological well-being.
 * 
 * @version 1.0.0
 * @size <2KB gzipped
 */

(function () {
  'use strict';

  // Time period definitions (24h format)
  const PERIODS = {
    MORNING: { start: 6, end: 11, name: 'morning' },
    AFTERNOON: { start: 12, end: 17, name: 'afternoon' },
    EVENING: { start: 18, end: 21, name: 'evening' },
    NIGHT: { start: 22, end: 5, name: 'night' }
  };

  // Cache for current period to avoid unnecessary DOM updates
  let currentPeriod = null;

  /**
   * Determine time period based on current hour
   * @param {number} hour - Current hour (0-23)
   * @returns {string} Period name
   */
  function getTimePeriod(hour) {
    if (hour >= PERIODS.MORNING.start && hour <= PERIODS.MORNING.end) {
      return PERIODS.MORNING.name;
    }
    if (hour >= PERIODS.AFTERNOON.start && hour <= PERIODS.AFTERNOON.end) {
      return PERIODS.AFTERNOON.name;
    }
    if (hour >= PERIODS.EVENING.start && hour <= PERIODS.EVENING.end) {
      return PERIODS.EVENING.name;
    }
    return PERIODS.NIGHT.name; // 22-23, 0-5
  }

  /**
   * Update data-time attribute on <html>
   * Only updates if period changed (avoids DOM thrashing)
   */
  function updateCircadianTime() {
    const now = new Date();
    const hour = now.getHours();
    const period = getTimePeriod(hour);

    // Skip if already set to current period
    if (period === currentPeriod) return;

    // Respect manually set data-time (for testing/debugging)
    const htmlElement = document.documentElement;
    const existingTime = htmlElement.getAttribute('data-time');
    
    // Don't override if manually set (not by this script)
    if (existingTime && existingTime !== currentPeriod) {
      console.debug('[CircadianSync] data-time manually set, skipping update');
      return;
    }

    htmlElement.setAttribute('data-time', period);
    currentPeriod = period;

    // Dispatch custom event for listeners
    window.dispatchEvent(
      new CustomEvent('circadian:period-change', {
        detail: { period, hour, timestamp: now.toISOString() }
      })
    );

    console.debug(`[CircadianSync] Period set to: ${period}`);
  }

  /**
   * Initialize the circadian sync
   * Uses requestAnimationFrame for efficient timing
   */
  function init() {
    // Set initial period
    updateCircadianTime();

    // Check every 60 seconds (60000ms)
    // Using setInterval is fine here — we're not animating, just checking time
    setInterval(updateCircadianTime, 60000);

    // Also listen for visibility changes (user returns to tab after hours)
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) {
        updateCircadianTime();
      }
    });

    console.debug('[CircadianSync] Initialized');
  }

  // Self-initialize on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM is already ready
    init();
  }

  // Expose API for manual control (optional)
  window.CircadianSync = {
    getPeriod: function () {
      return currentPeriod || getTimePeriod(new Date().getHours());
    },
    setPeriod: function (period) {
      if (['morning', 'afternoon', 'evening', 'night'].includes(period)) {
        document.documentElement.setAttribute('data-time', period);
        currentPeriod = period;
        console.debug(`[CircadianSync] Manually set to: ${period}`);
      }
    },
    refresh: updateCircadianTime
  };

})();
