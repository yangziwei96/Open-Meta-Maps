#!/usr/bin/env node
/**
 * Headless Chrome check: edit mode lock-labels + reset on a finished tool job.
 * Usage: node test_map_edit_chrome.js <baseUrl> <jobId> [screenshotPath]
 */
const fs = require('fs');
const path = require('path');
const pwPath = [
  path.join(__dirname, 'runtime/node_modules/playwright-core'),
  '/tmp/node_modules/playwright-core',
].find((p) => fs.existsSync(p));
const { chromium } = require(pwPath);

function almost(a, b, eps) {
  return Math.abs(a - b) <= (eps == null ? 0.6 : eps);
}

async function resultMapName(page, jobId) {
  return page.evaluate(async (id) => {
    const j = await (await fetch('/api/jobs/' + id)).json();
    if (j.summary && j.summary.result_basename) return j.summary.result_basename + '.json';
    const hit = (j.files || []).find((f) => /^OpenMetaMaps_.+_result\.json$/.test(f.name));
    if (!hit) throw new Error('result map JSON not listed in job files');
    return hit.name;
  }, jobId);
}

async function nodeLabel(page, nid) {
  return page.evaluate((id) => {
    const c = window.OPM_SVG.querySelector('circle[data-n="' + CSS.escape(id) + '"]');
    const t = window.OPM_SVG.querySelector('text[data-nl="' + CSS.escape(id) + '"]');
    if (!c || !t) return null;
    return {
      cx: +c.getAttribute('cx'),
      cy: +c.getAttribute('cy'),
      lx: +t.getAttribute('x'),
      ly: +t.getAttribute('y'),
    };
  }, nid);
}

async function dragBy(page, x, y, dx, dy) {
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(120);
}

async function screenCenter(page, kind, id) {
  return page.evaluate(
    ({ kind, id }) => {
      let el = null;
      if (kind === 'node') el = window.OPM_SVG.querySelector('circle[data-n="' + CSS.escape(id) + '"]');
      else if (kind === 'nlabel') el = window.OPM_SVG.querySelector('text[data-nl="' + CSS.escape(id) + '"]');
      else if (kind === 'sel') el = window.OPM_SVG.querySelector('circle.sel');
      if (!el) return null;
      const r = el.getBoundingClientRect();
      if (kind === 'nlabel') {
        // Prefer a point that actually hit-tests to this label (bbox center can sit under a neighbour).
        const trials = [
          [0.7, 0.5],
          [0.5, 0.5],
          [0.3, 0.5],
          [0.8, 0.4],
          [0.6, 0.6],
          [0.4, 0.4],
        ];
        for (const [fx, fy] of trials) {
          const x = r.left + r.width * fx;
          const y = r.top + r.height * fy;
          const top = document.elementFromPoint(x, y);
          if (top && top.getAttribute('data-nl') === id) return { x, y };
        }
      }
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    },
    { kind, id }
  );
}

async function main() {
  const base = process.argv[2] || 'http://127.0.0.1:8765';
  const jobId = process.argv[3];
  const shot = process.argv[4] || '';
  if (!jobId) {
    console.error('usage: node test_map_edit_chrome.js <baseUrl> <jobId> [screenshot]');
    process.exit(2);
  }
  const chrome =
    process.env.CHROME_PATH ||
    ['/usr/local/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium'].find((p) =>
      fs.existsSync(p)
    );
  const browser = await chromium.launch({
    executablePath: chrome,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(120000);
  // Accept confirm dialogs for reset
  page.on('dialog', async (d) => {
    await d.accept();
  });

  const url = `${base}/tool/index.html?job=${jobId}`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.OPM_SVG && window.OPM_PZ && window.OPM_EDIT, null, { timeout: 120000 });
  await page.waitForFunction(() => window.OPM_EDIT && window.OPM_EDIT.isReady && window.OPM_EDIT.isReady() && document.getElementById('pedit'), null, {
    timeout: 120000,
  });
  await page.waitForTimeout(400);

  const mapName = await resultMapName(page, jobId);
  const before = await page.evaluate(async ({id, mapName}) => {
    const r = await fetch('/api/jobs/' + id + '/files/' + mapName);
    const j = await r.json();
    const nid = Object.keys(j[1].nodes).find(
      (k) => j[1].nodes[k].node_type === 'metabolite' && j[1].nodes[k].label_x != null
    );
    const n = j[1].nodes[nid];
    let original = null;
    try {
      const o = await fetch('/api/jobs/' + id + '/files/map.original.json');
      if (o.ok) original = await o.json();
    } catch (e) {}
    return {
      nid,
      x: n.x,
      y: n.y,
      label_x: n.label_x,
      label_y: n.label_y,
      originalX: original ? original[1].nodes[nid].x : null,
      originalY: original ? original[1].nodes[nid].y : null,
      originalLX: original ? original[1].nodes[nid].label_x : null,
      originalLY: original ? original[1].nodes[nid].label_y : null,
      hadOriginal: !!original,
    };
  }, { id: jobId, mapName });
  if (!before.nid) throw new Error('no metabolite with label');

  await page.click('#pedit');
  await page.waitForFunction(() => window.OPM_EDIT.isActive());
  // Zoom toward the target so labels are hittable
  for (let i = 0; i < 5; i++) await page.click('#zin');
  await page.waitForTimeout(100);
  await page.evaluate((id) => {
    const c = window.OPM_SVG.querySelector('circle[data-n="' + CSS.escape(id) + '"]');
    if (c && window.OPM_PZ) {
      const x = +c.getAttribute('cx');
      const y = +c.getAttribute('cy');
      window.OPM_PZ.focus(x - 80, y - 80, 160, 160);
    }
  }, before.nid);
  await page.waitForTimeout(100);

  // Default: Lock labels ON
  const lockDefault = await page.evaluate(() => ({
    api: window.OPM_EDIT.getLockLabels(),
    aria: document.getElementById('plock').getAttribute('aria-pressed'),
  }));
  if (!lockDefault.api || lockDefault.aria !== 'true') {
    throw new Error('Lock labels should be ON by default: ' + JSON.stringify(lockDefault));
  }

  // --- Lock ON: node drag moves label by same delta ---
  const c0 = await screenCenter(page, 'node', before.nid);
  if (!c0) throw new Error('metabolite circle missing');
  const geom0 = await nodeLabel(page, before.nid);
  await dragBy(page, c0.x, c0.y, 50, 30);
  const geom1 = await nodeLabel(page, before.nid);
  const ndx = geom1.cx - geom0.cx;
  const ndy = geom1.cy - geom0.cy;
  const ldx = geom1.lx - geom0.lx;
  const ldy = geom1.ly - geom0.ly;
  if (Math.abs(ndx) < 1 && Math.abs(ndy) < 1) throw new Error('lock-on drag did not move node');
  if (!almost(ndx, ldx) || !almost(ndy, ldy)) {
    throw new Error('lock-on: label did not follow node: ' + JSON.stringify({ ndx, ndy, ldx, ldy }));
  }

  // Lock ON: clicking label selects the node (not independent label)
  await page.click('#pendo');
  await page.waitForTimeout(100);
  // Re-focus so the target label is the topmost hit
  await page.evaluate((id) => {
    const c = window.OPM_SVG.querySelector('circle[data-n="' + CSS.escape(id) + '"]');
    if (c && window.OPM_PZ) window.OPM_PZ.focus(+c.getAttribute('cx') - 60, +c.getAttribute('cy') - 60, 120, 120);
  }, before.nid);
  await page.waitForTimeout(80);
  const t0 = await screenCenter(page, 'nlabel', before.nid);
  if (!t0) throw new Error('metabolite label missing on screen');
  const hitAtLabel = await page.evaluate(({ x, y }) => {
    const el = document.elementFromPoint(x, y);
    return { tag: el && el.tagName, nl: el && el.getAttribute('data-nl'), n: el && el.getAttribute('data-n') };
  }, t0);
  if (hitAtLabel.nl !== before.nid) {
    throw new Error('could not find clear click point on label: ' + JSON.stringify({ hitAtLabel, expect: before.nid }));
  }
  await page.mouse.click(t0.x, t0.y);
  await page.waitForTimeout(80);
  const selLockOn = await page.evaluate((id) => ({
    nodeSel: !!window.OPM_SVG.querySelector('circle[data-n="' + CSS.escape(id) + '"].sel'),
    labelSel: !!window.OPM_SVG.querySelector('text[data-nl="' + CSS.escape(id) + '"].sel'),
  }), before.nid);
  if (!selLockOn.nodeSel || selLockOn.labelSel) {
    throw new Error('lock-on: label click should select node: ' + JSON.stringify({ selLockOn, hitAtLabel }));
  }

  // --- Lock OFF: node drag leaves label; label can drag alone ---
  await page.click('#plock');
  await page.waitForTimeout(50);
  const lockOff = await page.evaluate(() => window.OPM_EDIT.getLockLabels());
  if (lockOff) throw new Error('expected Lock labels OFF after toggle');

  const geom2 = await nodeLabel(page, before.nid);
  const c1 = await screenCenter(page, 'node', before.nid);
  await dragBy(page, c1.x, c1.y, 40, 20);
  const geom3 = await nodeLabel(page, before.nid);
  if (almost(geom3.cx, geom2.cx) && almost(geom3.cy, geom2.cy)) throw new Error('lock-off: node did not move');
  if (!almost(geom3.lx, geom2.lx) || !almost(geom3.ly, geom2.ly)) {
    throw new Error('lock-off: node move should not move label: ' + JSON.stringify({ geom2, geom3 }));
  }

  // Independent label drag
  const t1 = await screenCenter(page, 'nlabel', before.nid);
  await dragBy(page, t1.x, t1.y, 35, -20);
  const geom4 = await nodeLabel(page, before.nid);
  if (almost(geom4.lx, geom3.lx) && almost(geom4.ly, geom3.ly)) throw new Error('lock-off: label did not drag');
  if (!almost(geom4.cx, geom3.cx) || !almost(geom4.cy, geom3.cy)) {
    throw new Error('lock-off: label drag moved node: ' + JSON.stringify({ geom3, geom4 }));
  }

  // Undo restores previous geometry step
  await page.click('#pendo');
  await page.waitForTimeout(100);
  const geomUndo = await nodeLabel(page, before.nid);
  if (!almost(geomUndo.lx, geom3.lx) || !almost(geomUndo.ly, geom3.ly)) {
    throw new Error('undo did not restore label: ' + JSON.stringify({ geomUndo, geom3 }));
  }

  // Re-enable lock and verify node+label move together again
  await page.click('#plock');
  if (!(await page.evaluate(() => window.OPM_EDIT.getLockLabels()))) {
    throw new Error('failed to re-enable lock');
  }
  const geom5 = await nodeLabel(page, before.nid);
  const c2 = await screenCenter(page, 'node', before.nid);
  await dragBy(page, c2.x, c2.y, 30, 15);
  const geom6 = await nodeLabel(page, before.nid);
  if (!almost(geom6.cx - geom5.cx, geom6.lx - geom5.lx) || !almost(geom6.cy - geom5.cy, geom6.ly - geom5.ly)) {
    throw new Error('lock re-on: label sync failed');
  }

  // Box-select move with lock ON keeps label sync
  const box = await page.evaluate((id) => {
    const c = window.OPM_SVG.querySelector('circle[data-n="' + CSS.escape(id) + '"]');
    const r = c.getBoundingClientRect();
    return { x: r.left - 10, y: r.top - 10, x2: r.right + 10, y2: r.bottom + 10 };
  }, before.nid);
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x2, box.y2, { steps: 5 });
  await page.mouse.up();
  await page.waitForTimeout(80);
  const geom7 = await nodeLabel(page, before.nid);
  const mid = await screenCenter(page, 'sel', before.nid);
  if (!mid) throw new Error('box-select did not select node');
  await dragBy(page, mid.x, mid.y, 22, 12);
  const geom8 = await nodeLabel(page, before.nid);
  if (!almost(geom8.cx - geom7.cx, geom8.lx - geom7.lx) || !almost(geom8.cy - geom7.cy, geom8.ly - geom7.ly)) {
    throw new Error('lock-on box move: label sync failed');
  }

  if (shot) {
    await page.screenshot({ path: shot, fullPage: false });
  }

  // Save (large maps like iIT341 can take several minutes to regenerate SVG + quality)
  await page.click('#psave');
  await page.waitForFunction(() => !window.OPM_EDIT.isDirty(), null, { timeout: 600000 });
  await page.waitForTimeout(400);

  const saved = await page.evaluate(async ({ id, mapName }) => {
    const r = await fetch('/api/jobs/' + id + '/files/' + mapName, { cache: 'no-store' });
    return r.json();
  }, { id: jobId, mapName });
  const sn = saved[1].nodes[before.nid];
  if (almost(sn.x, before.x) && almost(sn.y, before.y)) {
    throw new Error('saved result map JSON positions unchanged');
  }
  // With lock on during last moves, relative offset should match current SVG
  const svgNow = await nodeLabel(page, before.nid);
  if (!almost(sn.x, svgNow.cx) || !almost(sn.label_x, svgNow.lx)) {
    throw new Error('saved positions mismatch SVG: ' + JSON.stringify({ sn, svgNow }));
  }

  // original kept
  const orig = await page.evaluate(async (id) => {
    const r = await fetch('/api/jobs/' + id + '/files/map.original.json');
    if (!r.ok) return null;
    return r.json();
  }, jobId);
  if (!orig) throw new Error('map.original.json missing after save');
  const ox = orig[1].nodes[before.nid].x;
  const oy = orig[1].nodes[before.nid].y;
  if (before.hadOriginal) {
    if (!almost(ox, before.originalX, 0.01) || !almost(oy, before.originalY, 0.01)) {
      throw new Error('original was overwritten');
    }
  } else if (!almost(ox, before.x, 0.01) || !almost(oy, before.y, 0.01)) {
    throw new Error('first original snapshot does not match pre-edit map');
  }

  // Reset button visible after original exists
  await page.waitForFunction(() => window.OPM_EDIT && window.OPM_EDIT.hasOriginal && window.OPM_EDIT.hasOriginal());
  const presetVisible = await page.evaluate(() => {
    const b = document.getElementById('preset');
    return b && !b.hidden && !b.disabled;
  });
  // may need edit mode on
  if (!(await page.evaluate(() => window.OPM_EDIT.isActive()))) {
    await page.click('#pedit');
    await page.waitForFunction(() => window.OPM_EDIT.isActive());
  }
  const presetOk = await page.evaluate(() => {
    const b = document.getElementById('preset');
    return b && !b.hidden && !b.disabled;
  });
  if (!presetOk && !presetVisible) throw new Error('Reset button should be available when original exists');

  // Reset restores original geometry (wait for a fresh editor instance after remount)
  const genBefore = await page.evaluate(() => (window.OPM_EDIT && window.OPM_EDIT.gen ? window.OPM_EDIT.gen() : 0));
  await page.click('#preset');
  await page.waitForFunction(
    (g) =>
      window.OPM_EDIT &&
      window.OPM_EDIT.isReady &&
      window.OPM_EDIT.isReady() &&
      window.OPM_EDIT.gen &&
      window.OPM_EDIT.gen() > g &&
      !window.OPM_EDIT.isActive() &&
      document.getElementById('pedit'),
    genBefore,
    { timeout: 600000 }
  );
  await page.waitForTimeout(300);
  const afterReset = await page.evaluate(async ({ id, mapName }) => {
    const r = await fetch('/api/jobs/' + id + '/files/' + mapName, { cache: 'no-store' });
    return r.json();
  }, { id: jobId, mapName });
  const rn = afterReset[1].nodes[before.nid];
  const expectX = before.hadOriginal ? before.originalX : before.x;
  const expectY = before.hadOriginal ? before.originalY : before.y;
  const expectLX = before.hadOriginal ? before.originalLX : before.label_x;
  const expectLY = before.hadOriginal ? before.originalLY : before.label_y;
  if (!almost(rn.x, expectX, 0.05) || !almost(rn.y, expectY, 0.05)) {
    throw new Error('reset did not restore node: ' + JSON.stringify({ rn, expectX, expectY }));
  }
  if (expectLX != null && (!almost(rn.label_x, expectLX, 0.05) || !almost(rn.label_y, expectLY, 0.05))) {
    throw new Error('reset did not restore label: ' + JSON.stringify({ rn, expectLX, expectLY }));
  }
  // Undo history cleared: undo disabled after re-entering edit
  await page.click('#pedit');
  await page.waitForFunction(() => window.OPM_EDIT && window.OPM_EDIT.isActive && window.OPM_EDIT.isActive());
  const undoDisabled = await page.evaluate(() => document.getElementById('pendo').disabled);
  if (!undoDisabled) throw new Error('undo should be disabled after reset (history cleared)');

  // Session lock preference remembered across remount
  const lockAfterReset = await page.evaluate(() => window.OPM_EDIT.getLockLabels());
  if (!lockAfterReset) throw new Error('lock preference should remain ON across remount');

  console.log(
    JSON.stringify(
      {
        ok: true,
        jobId,
        node: before.nid,
        before: { x: before.x, y: before.y, label_x: before.label_x, label_y: before.label_y },
        afterSave: { x: sn.x, y: sn.y, label_x: sn.label_x, label_y: sn.label_y },
        afterReset: { x: rn.x, y: rn.y, label_x: rn.label_x, label_y: rn.label_y },
      },
      null,
      2
    )
  );
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
