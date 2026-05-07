/**
 * Lighthouse Audit Script — Bio-Digital Minimalism 2026
 * 
 * Runs Lighthouse audits and outputs JSON report.
 * Requires: npm install -g lighthouse
 * 
 * Usage: node lighthouse-audit.js --target <url|file>
 */

const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const fs = require('fs');
const path = require('path');

// Parse CLI arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const config = { target: null, outputDir: './reports' };
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--target' && args[i + 1]) {
      config.target = args[i + 1];
      i++;
    }
    if (args[i] === '--output' && args[i + 1]) {
      config.outputDir = args[i + 1];
      i++;
    }
  }
  
  return config;
}

async function runLighthouse(target) {
  // Launch Chrome
  const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless', '--no-sandbox'] });
  
  const options = {
    logLevel: 'info',
    output: 'json',
    onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
    port: chrome.port
  };

  // Run Lighthouse
  const runnerResult = await lighthouse(target, options);
  const report = runnerResult.lhr;

  // Kill Chrome
  await chrome.kill();

  return report;
}

function generateReport(lighthouseResult, outputDir) {
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outputPath = path.join(outputDir, `lighthouse-${timestamp}.json`);
  
  // Write full JSON report
  fs.writeFileSync(outputPath, JSON.stringify(lighthouseResult, null, 2));
  
  // Also generate a summary
  const summary = {
    url: lighthouseResult.finalUrl,
    timestamp: lighthouseResult.fetchTime,
    scores: {
      performance: Math.round(lighthouseResult.categories.performance.score * 100),
      accessibility: Math.round(lighthouseResult.categories.accessibility.score * 100),
      bestPractices: Math.round(lighthouseResult.categories['best-practices'].score * 100),
      seo: Math.round(lighthouseResult.categories.seo.score * 100)
    },
    audits: {}
  };

  // Extract key audits
  const keyAudits = [
    'first-contentful-paint',
    'largest-contentful-paint',
    'total-blocking-time',
    'cumulative-layout-shift',
    'speed-index'
  ];

  keyAudits.forEach(auditId => {
    const audit = lighthouseResult.audits[auditId];
    if (audit) {
      summary.audits[auditId] = {
        title: audit.title,
        score: audit.score,
        displayValue: audit.displayValue
      };
    }
  });

  const summaryPath = path.join(outputDir, `lighthouse-summary-${timestamp}.json`);
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

  console.log('✅ Lighthouse audit complete!');
  console.log(`📊 Full report: ${outputPath}`);
  console.log(`📋 Summary: ${summaryPath}`);
  console.log('\nScores:');
  console.log(`  Performance: ${summary.scores.performance}/100`);
  console.log(`  Accessibility: ${summary.scores.accessibility}/100`);
  console.log(`  Best Practices: ${summary.scores.bestPractices}/100`);
  console.log(`  SEO: ${summary.scores.seo}/100`);

  return { outputPath, summaryPath, summary };
}

async function main() {
  const config = parseArgs();
  
  if (!config.target) {
    console.error('❌ Error: --target is required (URL or file path)');
    process.exit(1);
  }

  // Convert file path to file:// URL if needed
  let target = config.target;
  if (!target.startsWith('http') && fs.existsSync(target)) {
    target = 'file://' + path.resolve(target);
  }

  console.log(`🚀 Running Lighthouse audit for: ${target}`);
  
  try {
    const result = await runLighthouse(target);
    generateReport(result, config.outputDir);
  } catch (error) {
    console.error('❌ Lighthouse audit failed:', error.message);
    process.exit(1);
  }
}

main();
