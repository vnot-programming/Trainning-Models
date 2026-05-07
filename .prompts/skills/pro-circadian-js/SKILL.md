---
name: pro-circadian-js
description: 'Use when setting up circadian-sync color shifts for Bio-Digital Minimalism 2026. Provides a JS utility that sets data-time="morning|afternoon|evening|night" on <html> based on local time, enabling CSS circadian color palettes. Use when initializing new projects or adding biological well-being features.'
argument-hint: 'Optional: target HTML file to inject script into'
user-invocable: true
disable-model-invocation: false
---

# Circadian JS — Biological Time Sync Utility

You are a **Biological Well-being Specialist** implementing circadian-sync interfaces. Your job is to inject time-aware color shifts that align with human biological rhythms.

## What This Skill Does

Creates and injects a lightweight JavaScript utility (`circadian-sync.js`) that:
1. Detects the user's local time
2. Sets `data-time` attribute on `<html>` element
3. Updates automatically every minute (no interval thrashing)
4. Enables CSS circadian color palettes from the [pro-advanced-ui-ux](./skills/pro-advanced-ui-ux/) skill

## Time Periods

| Period | Hours | Biological Purpose |
|--------|-------|-------------------|
| `morning` | 6:00 AM - 11:59 AM | Higher contrast, cool tones for alertness |
| `afternoon` | 12:00 PM - 5:59 PM | Balanced contrast, neutral tones for focus |
| `evening` | 6:00 PM - 9:59 PM | Reduced contrast, warm tones for wind-down |
| `night` | 10:00 PM - 5:59 AM | Minimal blue light, warmest tones for sleep prep |

## Procedure

### Step 1: Copy the Utility Script

Copy [circadian-sync.js](./assets/circadian-sync.js) to your project's `assets/js/` or `public/js/` folder.

### Step 2: Inject into HTML

Add the script to your HTML `<head>` or before `</body>`:

```html
<script src="assets/js/circadian-sync.js" defer></script>
```

Or, if you provide a target HTML file, the skill will inject it automatically.

### Step 3: Verify CSS Custom Properties

Ensure your CSS uses the `data-time` attribute selectors from the [pro-advanced-ui-ux color system](./skills/pro-advanced-ui-ux/references/color-system.md):

```css
/* Morning (default) */
:root {
  --hue-primary: 220;
  --lightness-bg: 98%;
}

/* Evening — warmer */
[data-time="evening"] {
  --hue-primary: 30;
  --lightness-bg: 96%;
}

/* Night — warmest, lowest contrast */
[data-time="night"] {
  --hue-primary: 25;
  --lightness-bg: 95%;
}
```

## The Utility Script

The full source is in [assets/circadian-sync.js](./assets/circadian-sync.js). Key features:
- **Zero dependencies**: Pure vanilla JS
- **Lightweight**: <2KB minified
- **Self-initializing**: Runs on DOMContentLoaded
- **Auto-updates**: Checks time every 60 seconds
- **Respects user preference**: Won't override manually set `data-time`

## Verification

After setup, open browser DevTools and check:
```javascript
document.documentElement.getAttribute('data-time')
// Should return: "morning", "afternoon", "evening", or "night"
```

## Output

The skill will:
1. Copy `circadian-sync.js` to the target project
2. Inject the `<script>` tag into the specified HTML file (or provide the snippet)
3. Verify the CSS custom properties are properly set up
4. Report the current detected time period
