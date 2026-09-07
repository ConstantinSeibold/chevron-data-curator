// Runs the REAL progress-line block out of app.js against fabricated ticks.
//
// The thing under test is a judgement call, not markup: WHEN a job is called stalled. A fixed 20s
// is right for a download chunk and wrong for a SAM image, and a false "stalled" on a healthy run
// is worse than none — it teaches the user to ignore the warning that means something.
const fs = require("fs");

const escAttr = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function _fmtQty(n, unit){");
const end = src.indexOf("let _progTimer=null;");
if (start < 0 || end < 0) { console.log(JSON.stringify({ error: "could not locate the progress block" })); process.exit(0); }
eval(src.slice(start, end));

const tick = o => Object.assign(
  { phase: "p", done: 0, total: 0, unit: "", detail: "", rate: 0, eta: 0, elapsed: 0, stalled: 0,
    stall_after: 0, active: true }, o);
const isStalled = p => _progLine(p).includes("stalled:");

console.log(JSON.stringify({
  // a download that stopped: the 20s default still applies, and the advice is about the network
  deadDownload: {
    at10: isStalled(tick({ unit: "bytes", done: 1e6, total: 4e8, stalled: 10, rate: 1e5 })),
    at30: isStalled(tick({ unit: "bytes", done: 1e6, total: 4e8, stalled: 30, rate: 1e5 })),
    why: _progLine(tick({ unit: "bytes", done: 1e6, total: 4e8, stalled: 30, rate: 1e5 })),
  },
  // the first SAM image: nothing has ticked yet, so only the phase's own budget defends it
  firstImage: {
    at60: isStalled(tick({ unit: "images", done: 0, total: 80, stalled: 60, stall_after: 300 })),
    at400: isStalled(tick({ unit: "images", done: 0, total: 80, stalled: 400, stall_after: 300 })),
    why: _progLine(tick({ unit: "images", done: 0, total: 80, stalled: 400, stall_after: 300 })),
  },
  // a slow but healthy run: 1 image per 100s, quiet for 90s -> its own pace says it is fine
  slowPace: {
    at90: isStalled(tick({ unit: "images", done: 4, total: 80, stalled: 90, rate: 0.01 })),
    at400: isStalled(tick({ unit: "images", done: 4, total: 80, stalled: 400, rate: 0.01 })),
  },
  // loading a model reports bytes with no total: network advice would be a wrong diagnosis
  loading: _progLine(tick({ phase: "loading the model", unit: "bytes", stalled: 900, elapsed: 900 })),
  // the healthy line still carries rate and ETA, and the filename rides in `detail`
  healthy: _progLine(tick({ phase: "proposing (samhq_auto)", detail: "frame_100_endo.png",
                            unit: "images", done: 8, total: 80, rate: 0.05, eta: 1440 })),
}));
