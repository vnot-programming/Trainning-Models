/**
 * Animation Compliance Checker — Bio-Digital Minimalism 2026
 * 
 * Scans CSS/JS files for animation violations (non-60fps properties).
 * Only transform and opacity should be animated.
 * 
 * Usage: node animation-check.js --target <file|directory>
 */

const fs = require('fs');
const path = require('path');

// Forbidden CSS properties in animations/transitions
const FORBIDDEN_PROPERTIES = [
  'height', 'width', 'max-height', 'max-width', 'min-height', 'min-width',
  'top', 'left', 'right', 'bottom',
  'margin', 'margin-top', 'margin-bottom', 'margin-left', 'margin-right',
  'padding', 'padding-top', 'padding-bottom', 'padding-left', 'padding-right',
  'box-shadow', 'text-shadow',
  'background-color', 'background-position',
  'border-width', 'border-color'
];

// Regex patterns to detect violations
const PATTERNS = {
  transition: /transition\s*:\s*([^;]+)/gi,
  animation: /animation\s*:\s*([^;]+)/gi,
  keyframe: /@keyframes\s+([^{]+)\s*\{/gi,
  keyframeContent: /([\w-]+)\s*:\s*([^;}]+)/gi,
  willChange: /will-change\s*:\s*([^;]+)/gi,
  prefersReducedMotion: /@media\s*\(prefers-reduced-motion[^)]*\)/gi
};

// Severity levels
const SEVERITY = {
  HIGH: 'High',
  MEDIUM: 'Medium',
  LOW: 'Low'
};

function parseArgs() {
  const args = process.argv.slice(2);
  const config = { target: null, recursive: true };
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--target' && args[i + 1]) {
      config.target = args[i + 1];
      i++;
    }
    if (args[i] === '--no-recursive') {
      config.recursive = false;
    }
  }
  
  return config;
}

function findFiles(target, recursive = true) {
  const results = [];
  
  if (fs.statSync(target).isFile()) {
    return [target];
  }
  
  function walk(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory() && recursive) {
        // Skip node_modules, .git, etc.
        if (!['node_modules', '.git', 'dist', 'build', '.next'].includes(file)) {
          walk(fullPath);
        }
      } else if (stat.isFile() && /\.(css|scss|less|js|jsx|ts|tsx|vue|svelte)$/i.test(file)) {
        results.push(fullPath);
      }
    }
  }
  
  walk(target);
  return results;
}

function checkFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const violations = [];
  const lines = content.split('\n');
  
  // Check for forbidden properties in transitions
  let match;
  while ((match = PATTERNS.transition.exec(content)) !== null) {
    const transitionValue = match[1];
    FORBIDDEN_PROPERTIES.forEach(prop => {
      if (transitionValue.toLowerCase().includes(prop)) {
        const lineNum = content.substring(0, match.index).split('\n').length;
        violations.push({
          line: lineNum,
          property: prop,
          type: 'transition',
          value: transitionValue.trim(),
          severity: SEVERITY.HIGH
        });
      }
    });
  }
  
  // Check for forbidden properties in animations
  PATTERNS.animation.lastIndex = 0;
  while ((match = PATTERNS.animation.exec(content)) !== null) {
    const animationValue = match[1];
    FORBIDDEN_PROPERTIES.forEach(prop => {
      if (animationValue.toLowerCase().includes(prop)) {
        const lineNum = content.substring(0, match.index).split('\n').length;
        violations.push({
          line: lineNum,
          property: prop,
          type: 'animation',
          value: animationValue.trim(),
          severity: SEVERITY.HIGH
        });
      }
    });
  }
  
  // Check keyframes for forbidden properties
  const keyframeRegex = /@keyframes\s+([^{]+)\s*\{([^}]+)\}/gi;
  while ((match = keyframeRegex.exec(content)) !== null) {
    const keyframeName = match[1].trim();
    const keyframeBody = match[2];
    
    FORBIDDEN_PROPERTIES.forEach(prop => {
      const propRegex = new RegExp(`\\b${prop}\\s*:`, 'gi');
      if (propRegex.test(keyframeBody)) {
        const lineNum = content.substring(0, match.index).split('\n').length;
        violations.push({
          line: lineNum,
          property: prop,
          type: 'keyframe',
          value: `@keyframes ${keyframeName}`,
          severity: SEVERITY.HIGH
        });
      }
    });
  }
  
  // Check for missing will-change on animated elements
  const hasAnimations = /transition|animation/.test(content);
  if (hasAnimations) {
    PATTERNS.willChange.lastIndex = 0;
    if (!PATTERNS.willChange.test(content)) {
      violations.push({
        line: 0,
        property: 'will-change',
        type: 'missing',
        value: 'Missing will-change on animated elements',
        severity: SEVERITY.MEDIUM
      });
    }
  }
  
  // Check for missing prefers-reduced-motion
  PATTERNS.prefersReducedMotion.lastIndex = 0;
  if (hasAnimations && !PATTERNS.prefersReducedMotion.test(content)) {
    violations.push({
      line: 0,
      property: 'prefers-reduced-motion',
      type: 'missing',
      value: 'Missing @media (prefers-reduced-motion: reduce) handler',
      severity: SEVERITY.MEDIUM
    });
  }
  
  return violations;
}

function generateReport(allViolations) {
  const totalViolations = allViolations.reduce((sum, f) => sum + f.violations.length, 0);
  
  console.log('\n🎬 Animation Compliance Report — Bio-Digital Minimalism 2026\n');
  console.log(`Total violations: ${totalViolations}\n`);
  
  if (totalViolations === 0) {
    console.log('✅ No animation violations found! All animations are 60fps compliant.\n');
    return;
  }
  
  // Group by severity
  const bySeverity = { High: [], Medium: [], Low: [] };
  allViolations.forEach(file => {
    file.violations.forEach(v => {
      bySeverity[v.severity].push({ file: file.file, ...v });
    });
  });
  
  // Print High severity
  if (bySeverity.High.length > 0) {
    console.log('🔴 HIGH Severity (breaks 60fps):');
    console.log('| File | Line | Property | Type | Value |');
    console.log('|------|------|----------|------|-------|');
    bySeverity.High.forEach(v => {
      console.log(`| ${path.relative(process.cwd(), v.file)} | ${v.line || '-'} | ${v.property} | ${v.type} | ${v.value.substring(0, 30)}... |`);
    });
    console.log('');
  }
  
  // Print Medium severity
  if (bySeverity.Medium.length > 0) {
    console.log('🟡 MEDIUM Severity (should fix):');
    bySeverity.Medium.forEach(v => {
      console.log(`  - ${path.relative(process.cwd(), v.file)}: ${v.value}`);
    });
    console.log('');
  }
  
  // Print summary
  console.log('\n📊 Summary:');
  console.log(`  High: ${bySeverity.High.length}`);
  console.log(`  Medium: ${bySeverity.Medium.length}`);
  console.log(`  Low: ${bySeverity.Low.length}`);
  console.log('\n💡 Fix: Only animate \`transform\` and \`opacity\` for 60fps performance.');
  console.log('   Use \`will-change: transform, opacity\` on animated elements.');
  console.log('   Add \`@media (prefers-reduced-motion: reduce)\` handler.\n');
}

function main() {
  const config = parseArgs();
  
  if (!config.target) {
    console.error('❌ Error: --target is required (file or directory)');
    process.exit(1);
  }
  
  if (!fs.existsSync(config.target)) {
    console.error(`❌ Error: Target not found: ${config.target}`);
    process.exit(1);
  }
  
  console.log(`🔍 Scanning for animation violations in: ${config.target}`);
  
  const files = findFiles(config.target, config.recursive);
  const allViolations = [];
  
  files.forEach(file => {
    const violations = checkFile(file);
    if (violations.length > 0) {
      allViolations.push({ file, violations });
    }
  });
  
  generateReport(allViolations);
}

main();
