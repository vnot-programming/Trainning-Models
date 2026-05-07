# Bio-Digital Minimalism 2026 — Animation & Layout Guidelines

## Animation Principles (60fps+ Performance)

### The Only Two Properties Rule

For smooth 60fps+ animations, **only animate `transform` and `opacity`**. Never animate `height`, `width`, `top`, `left`, `margin`, `padding`, or `box-shadow` directly.

```css
/* ✅ GOOD — uses transform and opacity */
.fade-in {
  opacity: 0;
  transform: translateY(10px);
  transition:
    opacity 300ms ease,
    transform 300ms ease;
}

.fade-in.visible {
  opacity: 1;
  transform: translateY(0);
}

/* ❌ BAD — causes layout thrashing */
.slide-bad {
  transition:
    top 300ms ease,
    height 300ms ease; /* Never do this */
}
```

### Easing Curves — Biological Timing

Use cubic-bezier curves that mimic natural movement:

```css
:root {
  /* Standard easing — enters and exits naturally */
  --ease-natural: cubic-bezier(0.4, 0, 0.2, 1);

  /* Gentle — for hover states, subtle interactions */
  --ease-gentle: cubic-bezier(0.25, 0.46, 0.45, 0.94);

  /* Bounce — for delightful moments (use sparingly) */
  --ease-bounce: cubic-bezier(0.34, 1.56, 0.64, 1);

  /* Exit — things leaving the viewport */
  --ease-exit: cubic-bezier(0.55, 0.06, 0.68, 0.19);
}
```

### Duration Guidelines

- **Micro-interactions** (hover, focus): `150-200ms`
- **Standard transitions** (cards, modals): `300ms`
- **Page transitions**: `400-500ms`
- **Staggered lists**: `100ms` delay between items, `300ms` duration

### Will-Change Optimization

Only apply `will-change` to elements that are about to animate:

```css
/* ✅ Apply just before animation */
.element-animating {
  will-change: transform, opacity;
}

/* ❌ Don't blanket-apply */
* {
  will-change: transform, opacity; /* Kills performance */
}
```

## Layout System (Fluid & Responsive)

### Container — No Rigid Breakpoints

Use `clamp()` for fluid layouts that adapt naturally:

```css
/* Container — fluid width with max readability */
.container {
  width: 100%;
  max-width: 75ch; /* Optimal reading width */
  padding-inline: clamp(1rem, 5vw, 3rem);
  margin-inline: auto;
}

/* Wide container for dashboards */
.container-wide {
  width: 100%;
  max-width: clamp(60rem, 85vw, 90rem);
  padding-inline: clamp(1rem, 3vw, 2rem);
  margin-inline: auto;
}
```

### Grid — Fluid Columns

```css
/* Auto-fit grid — no media queries needed */
.grid-auto {
  display: grid;
  grid-template-columns: repeat(
    auto-fit,
    minmax(clamp(250px, 30%, 350px), 1fr)
  );
  gap: clamp(1rem, 3vw, 2rem);
}

/* 12-column fluid grid */
.grid-12 {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: clamp(0.75rem, 2vw, 1.5rem);
}

/* Responsive spanning */
.grid-span-full {
  grid-column: 1 / -1;
}
.grid-span-half {
  grid-column: span 6;
}
.grid-span-third {
  grid-column: span 4;
}

@media (max-width: 768px) {
  .grid-12 {
    grid-template-columns: 1fr;
  }
}
```

### Flexbox Patterns

```css
/* Stack — vertical rhythm */
.stack {
  display: flex;
  flex-direction: column;
  gap: clamp(0.5rem, 2vh, 1.5rem);
}

/* Cluster — wraps naturally */
.cluster {
  display: flex;
  flex-wrap: wrap;
  gap: clamp(0.5rem, 1.5vw, 1rem);
  align-items: center;
}

/* Center — perfect centering */
.center {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh; /* or specific height */
}
```

### Spacing Scale — Consistent Negative Space

```css
:root {
  /* Spacing — based on 1rem (16px) base */
  --space-3xs: clamp(0.25rem, 0.5vw, 0.375rem);
  --space-2xs: clamp(0.375rem, 0.75vw, 0.5rem);
  --space-xs: clamp(0.5rem, 1vw, 0.75rem);
  --space-sm: clamp(0.75rem, 1.5vw, 1rem);
  --space-md: clamp(1rem, 2vw, 1.5rem);
  --space-lg: clamp(1.5rem, 3vw, 2rem);
  --space-xl: clamp(2rem, 4vw, 3rem);
  --space-2xl: clamp(3rem, 6vw, 4rem);
  --space-3xl: clamp(4rem, 8vw, 6rem);
}
```

## Accessibility Requirements (WCAG 2.2+)

### Semantic HTML — Always

```html
<!-- ✅ Semantic -->
<article>
  <header>
    <h2>Article Title</h2>
    <time datetime="2026-05-04">May 4, 2026</time>
  </header>
  <p>Content here...</p>
</article>

<!-- ❌ Div soup -->
<div class="article">
  <div class="title">Article Title</div>
  <div class="content">Content here...</div>
</div>
```

### ARIA — Only When Necessary

```html
<!-- Use native elements when possible -->
<button class="btn">Save</button>
<!-- ✅ No ARIA needed -->

<!-- ARIA for custom components -->
<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="dialog-title"
  aria-describedby="dialog-desc"
>
  <h2 id="dialog-title">Confirm Action</h2>
  <p id="dialog-desc">Are you sure?</p>
</div>
```

### Focus Management

```css
/* Visible focus for keyboard users */
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 3px;
}

/* Remove focus for mouse users */
:focus:not(:focus-visible) {
  outline: none;
}
```

### Reduced Motion — Respect User Preferences

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

### Color Contrast — Programmatic Check

```css
/* Ensure contrast ratios meet WCAG 2.2 */
.text-on-primary {
  /* Primary bg: hsl(220, 12%, 45%) = #64748b */
  /* White text: #ffffff */
  /* Contrast ratio: 4.68:1 — WCAG AA for normal text */
  color: white;
  background: var(--color-primary);
}

.text-on-primary-lg {
  /* Large text (18pt+) needs only 3:1 */
  font-size: var(--text-lg);
  color: white;
  background: var(--color-primary);
}
```
