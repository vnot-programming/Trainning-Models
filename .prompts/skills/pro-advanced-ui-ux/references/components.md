# Bio-Digital Minimalism 2026 — Component System

## Button Component

```html
<!-- Primary Button -->
<button class="btn btn-primary">
  <span class="btn-label">Get Started</span>
</button>

<!-- Secondary Button -->
<button class="btn btn-secondary">
  <span class="btn-label">Learn More</span>
</button>

<!-- Ghost Button -->
<button class="btn btn-ghost">
  <span class="btn-label">Cancel</span>
</button>
```

```css
/* Button System — 60fps animations, biological comfort */
.btn {
  /* Reset */
  appearance: none;
  border: none;
  outline: none;
  cursor: pointer;
  text-decoration: none;

  /* Layout */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: 0.75em 1.5em;

  /* Typography */
  font-family: var(--font-sans);
  font-weight: var(--weight-medium);
  font-size: var(--text-base);
  line-height: var(--leading-none);
  letter-spacing: var(--tracking-wide);

  /* Shape — generous border-radius for organic feel */
  border-radius: 0.75em;

  /* Animation — only transform and opacity */
  transition:
    transform 200ms cubic-bezier(0.4, 0, 0.2, 1),
    opacity 200ms cubic-bezier(0.4, 0, 0.2, 1),
    box-shadow 200ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: transform, opacity;

  /* Prevent layout shifts */
  backface-visibility: hidden;
  -webkit-font-smoothing: antialiased;
}

/* Primary — low saturation, premium feel */
.btn-primary {
  background: var(--color-primary);
  color: white;
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.08),
    0 4px 12px rgba(0, 0, 0, 0.04);
}

.btn-primary:hover {
  background: var(--color-primary-hover);
  transform: translateY(-1px);
  box-shadow:
    0 2px 6px rgba(0, 0, 0, 0.1),
    0 8px 24px rgba(0, 0, 0, 0.06);
}

.btn-primary:active {
  transform: translateY(0);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
}

/* Secondary — glassmorphism */
.btn-secondary {
  background: var(--glass-bg);
  color: var(--color-text-primary);
  border: 1px solid var(--glass-border);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: var(--glass-shadow);
}

.btn-secondary:hover {
  transform: translateY(-1px);
  box-shadow:
    0 2px 6px rgba(0, 0, 0, 0.08),
    0 8px 24px rgba(0, 0, 0, 0.04);
}

/* Ghost — minimal */
.btn-ghost {
  background: transparent;
  color: var(--color-text-secondary);
  padding: 0.75em 1em;
}

.btn-ghost:hover {
  color: var(--color-text-primary);
  background: rgba(0, 0, 0, 0.04);
}

@media (prefers-color-scheme: dark) {
  .btn-ghost:hover {
    background: rgba(255, 255, 255, 0.06);
  }
}

/* Disabled state */
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none !important;
}

/* Focus — accessible, not aggressive */
.btn:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 3px;
}

/* Size variants */
.btn-sm {
  padding: 0.5em 1em;
  font-size: var(--text-sm);
  border-radius: 0.625em;
}

.btn-lg {
  padding: 1em 2em;
  font-size: var(--text-lg);
  border-radius: 1em;
}
```

## Card Component

```html
<article class="card">
  <div class="card-media">
    <img src="image.jpg" alt="Description" class="card-image" />
  </div>
  <div class="card-body">
    <span class="card-label text-caption">Category</span>
    <h3 class="card-title text-h3">Card Title</h3>
    <p class="card-description text-body-sm">
      Brief description that serves a purpose. No lorem ipsum.
    </p>
    <div class="card-actions">
      <button class="btn btn-primary btn-sm">Read More</button>
    </div>
  </div>
</article>
```

```css
/* Card System — multi-layer shadows, expansive negative space */
.card {
  /* Layout */
  display: flex;
  flex-direction: column;
  overflow: hidden;

  /* Spacing — generous negative space */
  gap: 0;

  /* Surface */
  background: var(--color-bg-surface);
  border-radius: 1.25em;
  border: 1px solid rgba(0, 0, 0, 0.04);

  /* Multi-layer shadow — depth without harshness */
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.02),
    0 4px 12px rgba(0, 0, 0, 0.02),
    0 16px 32px rgba(0, 0, 0, 0.01);

  /* Animation */
  transition:
    transform 300ms cubic-bezier(0.4, 0, 0.2, 1),
    box-shadow 300ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: transform, box-shadow;
}

.card:hover {
  transform: translateY(-4px);
  box-shadow:
    0 2px 6px rgba(0, 0, 0, 0.04),
    0 8px 24px rgba(0, 0, 0, 0.03),
    0 24px 48px rgba(0, 0, 0, 0.02);
}

/* Card media */
.card-media {
  aspect-ratio: 16 / 9;
  overflow: hidden;
}

.card-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 500ms cubic-bezier(0.4, 0, 0.2, 1);
  will-change: transform;
}

.card:hover .card-image {
  transform: scale(1.05);
}

/* Card body — expansive padding */
.card-body {
  display: flex;
  flex-direction: column;
  gap: 0.75em;
  padding: 1.5em 1.75em 1.75em;
}

.card-label {
  text-transform: uppercase;
  letter-spacing: var(--tracking-widest);
  color: var(--color-text-tertiary);
}

.card-title {
  margin: 0;
  line-height: var(--leading-tight);
}

.card-description {
  color: var(--color-text-secondary);
  margin: 0;
}

.card-actions {
  display: flex;
  gap: 0.75em;
  margin-top: 0.5em;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
  .card {
    border-color: rgba(255, 255, 255, 0.06);
    box-shadow:
      0 1px 3px rgba(0, 0, 0, 0.2),
      0 4px 12px rgba(0, 0, 0, 0.15),
      0 16px 32px rgba(0, 0, 0, 0.1);
  }

  .card:hover {
    box-shadow:
      0 2px 6px rgba(0, 0, 0, 0.25),
      0 8px 24px rgba(0, 0, 0, 0.2),
      0 24px 48px rgba(0, 0, 0, 0.15);
  }
}
```

## Input Component

```html
<div class="input-group">
  <label for="email" class="input-label">Email Address</label>
  <input
    type="email"
    id="email"
    class="input"
    placeholder="you@example.com"
    autocomplete="email"
  />
  <span class="input-helper">We'll never share your email.</span>
</div>
```

```css
/* Input System — calm, focused, accessible */
.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.5em;
}

.input-label {
  font-family: var(--font-sans);
  font-weight: var(--weight-medium);
  font-size: var(--text-sm);
  color: var(--color-text-primary);
  letter-spacing: var(--tracking-wide);
}

.input {
  /* Reset */
  appearance: none;
  border: none;
  outline: none;
  background: none;

  /* Layout */
  width: 100%;
  padding: 0.875em 1em;

  /* Typography */
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: var(--leading-normal);
  color: var(--color-text-primary);

  /* Surface */
  background: var(--color-bg-surface);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 0.75em;

  /* Transition */
  transition:
    border-color 200ms ease,
    box-shadow 200ms ease;
}

.input::placeholder {
  color: var(--color-text-tertiary);
  opacity: 1;
}

.input:hover {
  border-color: rgba(0, 0, 0, 0.12);
}

.input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(var(--hue-primary), 50%, 50%, 0.1);
}

.input:invalid:not(:focus):not(:placeholder-shown) {
  border-color: var(--color-error);
  box-shadow: 0 0 0 3px rgba(5, 50%, 50%, 0.08);
}

.input-helper {
  font-size: var(--text-xs);
  color: var(--color-text-tertiary);
  letter-spacing: var(--tracking-wide);
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
  .input {
    border-color: rgba(255, 255, 255, 0.1);
    background: var(--color-bg-elevated);
  }

  .input:hover {
    border-color: rgba(255, 255, 255, 0.15);
  }

  .input:focus {
    border-color: var(--color-primary);
    box-shadow: 0 0 0 3px rgba(var(--hue-primary), 50%, 60%, 0.15);
  }
}
```
