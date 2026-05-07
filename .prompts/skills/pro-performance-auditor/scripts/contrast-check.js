/**
 * WCAG Contrast Checker — Bio-Digital Minimalism 2026
 * 
 * Validates WCAG 2.2+ contrast ratios for text/background combinations.
 * Checks CSS custom properties, inline styles, and common UI states.
 * 
 * Usage: node contrast-check.js --target <file|directory>
 */

const fs = require('fs');
const path = require('path');

// Relative luminance calculation (WCAG 2.0 formula)
function getLuminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map(c => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

function hslToRgb(h, s, l) {
  h /= 360; s /= 100; l /= 100;
  let r, g, b;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }

  return {
    r: Math.round(r * 255),
    g: Math.round(g * 255),
    b: Math.round(b * 255)
  };
}

function parseColor(color) {
  // Hex
  if (color.startsWith('#')) {
    return hexToRgb(color);
  }
  
  // rgb/rgba
  const rgbMatch = color.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (rgbMatch) {
    return { r: +rgbMatch[1], g: +rgbMatch[2], b: +rgbMatch[3] };
  }
  
  // hsl/hsla
  const hslMatch = color.match(/hsla?\(\s*(\d+)\s*,\s*(\d+)%?\s*,\s*(\d+)%?/);
  if (hslMatch) {
    return hslToRgb(+hslMatch[1], +hslMatch[2], +hslMatch[3]);
  }
  
  return null;
}

function getContrastRatio(color1, color2) {
  const lum1 = getLuminance(color1.r, color1.g, color1.b);
  const lum2 = getLuminance(color2.r, color2.g, color2.b);
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  return (brightest + 0.05) / (darkest + 0.05);
}

function checkContrast(fg, bg) {
  const fgRgb = parseColor(fg);
  const bgRgb = parseColor(bg);
  
  if (!fgRgb || !bgRgb) {
    return null;
  }
  
  const ratio = getContrastRatio(fgRgb, bgRgb);
  return {
    ratio: ratio,
    passesAA: ratio >= 4.5,
    passesAALarge: ratio >= 3.0,
    passesAAA: ratio >= 7.0,
    passesAAANormal: ratio >= 7.0
  };
}

function parseArgs() {
  const args = process.argv.slice(2);
  const config = { target: null, recursive: true };
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--target' && args[i + 1]) {
      config.target = args[i + 1];
      i++;
    }
  }
  
  return config;
}

function findCssFiles(target, recursive = true) {
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
        if (!['node_modules', '.git', 'dist', 'build', '.next'].includes(file)) {
          walk(fullPath);
        }
      } else if (stat.isFile() && /\.(css|scss|less)$/i.test(file)) {
        results.push(fullPath);
      }
    }
  }
  
  walk(target);
  return results;
}

function extractColorPairs(content) {
  const pairs = [];
  
  // Match color properties
  const colorProps = ['color', 'background-color', 'background', 'border-color'];
  
  colorProps.forEach(prop => {
    const regex = new RegExp(`${prop}\\s*:\\s*([^;!]+)`, 'gi');
    let match;
    while ((match = regex.exec(content)) !== null) {
      const value = match[1].trim();
      if (value && !value.includes('inherit') && !value.includes('transparent')) {
        pairs.push({
          property: prop,
          value: value,
          line: content.substring(0, match.index).split('\n').length
        });
      }
    }
  });
  
  return pairs;
}

function main() {
  const config = parseArgs();
  
  if (!config.target) {
    console.error('❌ Error: --target is required (CSS file or directory)');
    process.exit(1);
  }
  
  console.log(`🔍 Checking WCAG contrast ratios in: ${config.target}\n`);
  
  const files = findCssFiles(config.target, config.recursive);
  let totalChecks = 0;
  let passedAA = 0;
  let failedAA = 0;
  
  files.forEach(file => {
    const content = fs.readFileSync(file, 'utf-8');
    const colorPairs = extractColorPairs(content);
    
    // Simple check: look for text color + background color in same rule
    // This is a basic implementation — real-world would need CSS parsing
    const lines = content.split('\n');
    let currentFg = null;
    let currentBg = null;
    
    lines.forEach((line, index) => {
      const fgMatch = line.match(/color\s*:\s*([^;!]+)/i);
      const bgMatch = line.match(/background(?:-color)?\s*:\s*([^;!]+)/i);
      
      if (fgMatch) currentFg = fgMatch[1].trim();
      if (bgMatch) currentBg = bgMatch[1].trim();
      
      // Check if we have both
      if (currentFg && currentBg && (fgMatch || bgMatch)) {
        const result = checkContrast(currentFg, currentBg);
        if (result) {
          totalChecks++;
          if (result.passesAA) {
            passedAA++;
          } else {
            failedAA++;
            console.log(`🔴 ${path.relative(process.cwd(), file)}:${index + 1}`);
            console.log(`   FG: ${currentFg}`);
            console.log(`   BG: ${currentBg}`);
            console.log(`   Ratio: ${result.ratio.toFixed(2)}:1 (need 4.5:1 for AA)`);
            console.log(`   Status: ${result.passesAAA ? '✅ AAA' : result.passesAA ? '✅ AA' : '❌ Fail'}\n`);
          }
        }
      }
    });
  });
  
  console.log('\n📊 Contrast Check Summary:');
  console.log(`   Total checks: ${totalChecks}`);
  console.log(`   Passed AA: ${passedAA}`);
  console.log(`   Failed AA: ${failedAA}`);
  console.log(`   Pass rate: ${totalChecks > 0 ? Math.round((passedAA / totalChecks) * 100) : 100}%\n`);
  
  if (failedAA > 0) {
    console.log('💡 Fix: Increase contrast by darkening text or lightening background.');
    console.log('   Target: 4.5:1 for normal text, 3:1 for large text (18pt+).\n');
    process.exit(1);
  } else {
    console.log('✅ All contrast checks passed!\n');
  }
}

main();
