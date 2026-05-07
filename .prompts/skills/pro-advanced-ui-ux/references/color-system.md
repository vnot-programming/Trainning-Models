# Bio-Digital Minimalism 2026 — Color System

## Circadian-Sync Contrast System

The interface adapts to the user's biological time of day using CSS custom properties that shift based on the `data-theme` attribute or system preferences.

### Core Principle

- **Morning (6AM-12PM)**: Higher contrast, cooler tones (blue-shifted) to promote alertness
- **Afternoon (12PM-6PM)**: Balanced contrast, neutral tones for sustained focus
- **Evening (6PM-10PM)**: Reduced contrast, warmer tones (amber-shifted) to prepare for rest
- **Night (10PM-6AM)**: Minimal blue light, very warm tones, lowest contrast

### CSS Custom Properties — Light Mode

```css
:root {
  /* Base palette — HSL format for programmatic adjustment */
  --hue-primary: 220;
  --hue-secondary: 180;
  --hue-accent: 340;

  /* Circadian-sync base values (morning default) */
  --saturation-base: 12%;
  --lightness-bg: 98%;
  --lightness-surface: 95%;
  --lightness-text: 15%;
  --contrast-ratio: 7; /* WCAG AAA target */

  /* Background layers */
  --color-bg-primary: hsl(var(--hue-primary), 8%, var(--lightness-bg));
  --color-bg-surface: hsl(var(--hue-primary), 6%, var(--lightness-surface));
  --color-bg-elevated: hsl(var(--hue-primary), 5%, 100%);

  /* Text hierarchy */
  --color-text-primary: hsl(var(--hue-primary), 15%, var(--lightness-text));
  --color-text-secondary: hsl(var(--hue-primary), 10%, 35%);
  --color-text-tertiary: hsl(var(--hue-primary), 8%, 55%);

  /* Accent colors — low saturation for biological comfort */
  --color-primary: hsl(var(--hue-primary), var(--saturation-base), 45%);
  --color-primary-hover: hsl(
    var(--hue-primary),
    calc(var(--saturation-base) + 5%),
    40%
  );
  --color-secondary: hsl(var(--hue-secondary), var(--saturation-base), 45%);
  --color-accent: hsl(var(--hue-accent), var(--saturation-base), 50%);

  /* Semantic colors — muted, not alarmist */
  --color-success: hsl(150, 10%, 45%);
  --color-warning: hsl(35, 12%, 50%);
  --color-error: hsl(5, 12%, 55%);
  --color-info: hsl(var(--hue-primary), 10%, 50%);

  /* Glassmorphism */
  --glass-bg: rgba(255, 255, 255, 0.6);
  --glass-border: rgba(255, 255, 255, 0.3);
  --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
}

/* Afternoon — balanced */
[data-time="afternoon"] {
  --saturation-base: 10%;
  --lightness-bg: 97%;
  --contrast-ratio: 6.5;
}

/* Evening — warmer, reduced contrast */
[data-time="evening"] {
  --hue-primary: 30;
  --saturation-base: 8%;
  --lightness-bg: 96%;
  --lightness-text: 18%;
  --contrast-ratio: 5.5;
}

/* Night — minimal blue, warmest */
[data-time="night"] {
  --hue-primary: 25;
  --saturation-base: 6%;
  --lightness-bg: 95%;
  --lightness-text: 20%;
  --contrast-ratio: 4.5; /* WCAG AA minimum for night */
}
```

### CSS Custom Properties — Dark Mode

```css
@media (prefers-color-scheme: dark) {
  :root {
    --saturation-base: 15%;
    --lightness-bg: 8%;
    --lightness-surface: 12%;
    --lightness-text: 90%;

    --color-bg-primary: hsl(var(--hue-primary), 12%, var(--lightness-bg));
    --color-bg-surface: hsl(var(--hue-primary), 10%, var(--lightness-surface));
    --color-bg-elevated: hsl(var(--hue-primary), 8%, 16%);

    --color-text-primary: hsl(var(--hue-primary), 10%, var(--lightness-text));
    --color-text-secondary: hsl(var(--hue-primary), 8%, 75%);
    --color-text-tertiary: hsl(var(--hue-primary), 6%, 55%);

    --color-primary: hsl(var(--hue-primary), var(--saturation-base), 65%);
    --color-primary-hover: hsl(
      var(--hue-primary),
      calc(var(--saturation-base) + 5%),
      70%
    );

    --glass-bg: rgba(20, 20, 30, 0.6);
    --glass-border: rgba(255, 255, 255, 0.1);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }

  /* Evening dark — warmer */
  [data-time="evening"] {
    --hue-primary: 25;
    --lightness-bg: 10%;
    --lightness-text: 88%;
  }

  /* Night dark — warmest, lowest contrast */
  [data-time="night"] {
    --hue-primary: 20;
    --saturation-base: 10%;
    --lightness-bg: 12%;
    --lightness-text: 85%;
    --contrast-ratio: 4.5;
  }
}
```

## Harmonious Palette Selection

### Primary Hue Selection by Use Case

- **220 (Blue)**: Productivity apps, dashboards, professional tools
- **180 (Teal)**: Health, wellness, meditation apps
- **150 (Green)**: Finance, growth, environmental apps
- **30 (Warm)**: Creative tools, evening-use apps
- **260 (Purple)**: Premium services, luxury brands

### Saturation Rules

- **Backgrounds**: 5-10% — nearly neutral, biologically calming
- **Surfaces**: 6-12% — subtle color identity
- **Interactive elements**: 10-15% — visible but not aggressive
- **Never exceed 20%** for any UI element — prevents overstimulation

### Lightness Contrast Ratios

- **Text on surface**: Minimum 4.5:1 (WCAG AA), target 7:1 (WCAG AAA)
- **Large text (18pt+)**: Minimum 3:1 (WCAG AA)
- **Non-text elements**: Minimum 3:1 for boundaries and states
