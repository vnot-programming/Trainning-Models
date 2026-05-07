# Bio-Digital Minimalism 2026 — Typography System

## Font Stack

```css
/* Import from Google Fonts — optimal for biological readability */
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap");

:root {
  /* Font families */
  --font-sans:
    "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-display: "Outfit", var(--font-sans);
  --font-serif: "Lora", "Georgia", serif;

  /* Base size — fluid using clamp() */
  --text-base: clamp(1rem, 0.95rem + 0.25vw, 1.125rem);
  --text-scale: 1.25; /* Minor third scale */

  /* Type scale */
  --text-xs: clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem);
  --text-sm: clamp(0.875rem, 0.825rem + 0.25vw, 1rem);
  --text-base: clamp(1rem, 0.95rem + 0.25vw, 1.125rem);
  --text-lg: clamp(1.125rem, 1.05rem + 0.375vw, 1.375rem);
  --text-xl: clamp(1.375rem, 1.25rem + 0.625vw, 1.875rem);
  --text-2xl: clamp(1.875rem, 1.625rem + 1.25vw, 2.5rem);
  --text-3xl: clamp(2.5rem, 2rem + 2.5vw, 3.75rem);

  /* Line heights — optimized for cognitive ease */
  --leading-none: 1;
  --leading-tight: 1.25;
  --leading-snug: 1.375;
  --leading-normal: 1.5;
  --leading-relaxed: 1.625;
  --leading-loose: 2;

  /* Letter spacing — open for readability */
  --tracking-tighter: -0.05em;
  --tracking-tight: -0.025em;
  --tracking-normal: 0;
  --tracking-wide: 0.025em;
  --tracking-wider: 0.05em;
  --tracking-widest: 0.1em;

  /* Font weights */
  --weight-light: 300;
  --weight-normal: 400;
  --weight-medium: 500;
  --weight-semibold: 600;
  --weight-bold: 700;
}
```

## Typography Hierarchy

```css
/* Display — Outfit for impact without aggression */
.text-display {
  font-family: var(--font-display);
  font-weight: var(--weight-semibold);
  font-size: var(--text-3xl);
  line-height: var(--leading-tight);
  letter-spacing: var(--tracking-tight);
  color: var(--color-text-primary);
}

/* Headings */
.text-h1 {
  font-family: var(--font-display);
  font-weight: var(--weight-semibold);
  font-size: var(--text-2xl);
  line-height: var(--leading-tight);
  letter-spacing: var(--tracking-tight);
}

.text-h2 {
  font-family: var(--font-display);
  font-weight: var(--weight-medium);
  font-size: var(--text-xl);
  line-height: var(--leading-snug);
}

.text-h3 {
  font-family: var(--font-display);
  font-weight: var(--weight-medium);
  font-size: var(--text-lg);
  line-height: var(--leading-snug);
}

/* Body — Inter for sustained reading */
.text-body {
  font-family: var(--font-sans);
  font-weight: var(--weight-normal);
  font-size: var(--text-base);
  line-height: var(--leading-relaxed);
  letter-spacing: var(--tracking-normal);
  color: var(--color-text-primary);
}

.text-body-sm {
  font-size: var(--text-sm);
  line-height: var(--leading-relaxed);
}

.text-body-lg {
  font-size: var(--text-lg);
  line-height: var(--leading-relaxed);
}

/* Caption — Lora italic for elegance */
.text-caption {
  font-family: var(--font-serif);
  font-style: italic;
  font-size: var(--text-sm);
  line-height: var(--leading-normal);
  color: var(--color-text-secondary);
  letter-spacing: var(--tracking-wide);
}

/* Code — monospace with generous spacing */
.text-code {
  font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  font-size: var(--text-sm);
  line-height: var(--leading-relaxed);
  letter-spacing: var(--tracking-normal);
}
```

## Biological Readability Guidelines

### Cognitive Load Reduction

- **Line length**: Maximum 70 characters for body text (measure with `ch` units)
- **Paragraph spacing**: `margin-bottom: 1.5em` for comfortable scanning
- **No justified text**: Always left-aligned — ragged right reduces cognitive strain

### Hierarchy Clarity

- Use weight and size for hierarchy, not color alone
- Maximum 3 font sizes visible at once in any viewport
- Display font (Outfit) reserved for hero sections only

### Dark Mode Adjustments

```css
@media (prefers-color-scheme: dark) {
  :root {
    --font-sans:
      "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    /* Slightly heavier weights in dark mode for readability */
    --weight-normal: 400; /* unchanged */
    --weight-medium: 500; /* unchanged */
  }
}
```
