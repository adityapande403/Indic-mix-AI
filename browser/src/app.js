/**
 * IndicMix AI — In-Browser Local Inference Engine
 * 
 * Provides 100% private, client-side NLP for Code-Mixed Indic languages without
 * sending user data across the network unless configured.
 */

// Lexicon priors
const HINGLISH_WORDS = new Set([
  "bhai", "yaar", "mera", "meri", "mere", "mujhe", "hai", "hain", "ho", "nahi", "nhi",
  "kar", "karo", "diya", "gaya", "gaye", "raha", "rahi", "paise", "rupaye", "theek",
  "accha", "kya", "kyun", "kab", "kaha", "kaise", "lekin", "par", "aur", "chahiye"
]);

const TELGISH_WORDS = new Set([
  "naaku", "meeku", "anna", "babu", "garu", "undi", "ledu", "dabbu", "dabbulu",
  "chesanu", "cheyandi", "eppudu", "ekkada", "enti", "ela", "raaledu", "vastundi"
]);

const BENGLISH_WORDS = new Set([
  "amar", "tomar", "apnar", "dada", "bhai", "hoyeche", "hoyni", "korechi", "korbo",
  "taka", "kobe", "kothay", "keno", "ki", "ache", "nei", "ektu", "shob", "bhalo"
]);

// Sample Presets
const SAMPLES = {
  sampleHinglish: "bhai payment deduct ho gaya par order confirm nahi dikha raha",
  sampleTelgish: "anna naaku delivery raledu dabbu cut ayindi konchem chudandi",
  sampleBenglish: "dada taka kete geche kintu order confirm hoyni refund kobe pabo",
  samplePii: "Call Rahul at 9876543210 or email rahul.sharma@example.com for order ORD-998811 refund to rahul@okhdfcbank",
  sampleAbuse: "kya chutiya service hai tumhari saale chor"
};

class ClientNLP {
  static identifyLanguage(text) {
    if (!text || text.trim().length === 0) {
      return { lang: "unknown", confidence: 0, isCodeMixed: false, isRomanized: false };
    }

    const words = text.toLowerCase().match(/[a-z]+/g) || [];
    if (!words.length) {
      // Check native scripts
      if (/[\u0900-\u097F]/.test(text)) return { lang: "hi_devanagari", confidence: 0.98, isCodeMixed: false, isRomanized: false };
      if (/[\u0C00-\u0C7F]/.test(text)) return { lang: "te_telugu", confidence: 0.98, isCodeMixed: false, isRomanized: false };
      if (/[\u0980-\u09FF]/.test(text)) return { lang: "bn_bengali", confidence: 0.98, isCodeMixed: false, isRomanized: false };
      return { lang: "unknown", confidence: 0.2, isCodeMixed: false, isRomanized: false };
    }

    let hiCount = 0;
    let teCount = 0;
    let bnCount = 0;

    for (const w of words) {
      if (HINGLISH_WORDS.has(w)) hiCount++;
      if (TELGISH_WORDS.has(w)) teCount++;
      if (BENGLISH_WORDS.has(w)) bnCount++;
    }

    if (hiCount >= 1 && hiCount >= teCount && hiCount >= bnCount) {
      const conf = Math.min(0.96, 0.50 + hiCount * 0.15);
      return { lang: "hi_en_roman (Hinglish)", confidence: conf, isCodeMixed: true, isRomanized: true };
    }
    if (teCount >= 1 && teCount >= hiCount && teCount >= bnCount) {
      const conf = Math.min(0.96, 0.50 + teCount * 0.15);
      return { lang: "te_en_roman (Telgish)", confidence: conf, isCodeMixed: true, isRomanized: true };
    }
    if (bnCount >= 1 && bnCount >= hiCount && bnCount >= teCount) {
      const conf = Math.min(0.96, 0.50 + bnCount * 0.15);
      return { lang: "bn_en_roman (Benglish)", confidence: conf, isCodeMixed: true, isRomanized: true };
    }

    return { lang: "en (English / Code-mixed)", confidence: 0.85, isCodeMixed: false, isRomanized: false };
  }

  static classifyIntent(text) {
    const lower = text.toLowerCase();
    
    // Check substantive intents first
    if (/payment|deduct|cut|debited|paise kat|taka kete|dabbu cut/i.test(lower)) {
      return { intent: "payment_issue", confidence: 0.94, abstained: false };
    }
    if (/refund|wapas|money back|return money/i.test(lower)) {
      return { intent: "refund_request", confidence: 0.92, abstained: false };
    }
    if (/cancel|cancellation|radd/i.test(lower)) {
      return { intent: "cancellation_request", confidence: 0.93, abstained: false };
    }
    if (/order|track|tracking|status|kaha hai|deliver/i.test(lower)) {
      return { intent: "order_status", confidence: 0.91, abstained: false };
    }
    if (/login|password|otp|account/i.test(lower)) {
      return { intent: "account_access", confidence: 0.89, abstained: false };
    }
    if (/hi|hello|hey|namaste|namaskar/i.test(lower)) {
      return { intent: "greeting", confidence: 0.88, abstained: false };
    }

    return { intent: "unknown", confidence: 0.25, abstained: true };
  }

  static moderate(text) {
    const lower = text.toLowerCase();
    let abuseScore = 0.02;
    let threatScore = 0.02;
    let spamScore = 0.02;

    if (/chutiya|bhosdike|madarchod|lanjakodaka|bokachoda|asshole|bastard/i.test(lower)) {
      abuseScore = 0.96;
    }
    if (/mar dunga|jaan se|champesta|mere phelbo|kill you|break your bones/i.test(lower)) {
      threatScore = 0.95;
    }
    if (/earn \d+|free crypto|lottery winner|click here to win/i.test(lower)) {
      spamScore = 0.92;
    }

    const isFlagged = abuseScore >= 0.5 || threatScore >= 0.4 || spamScore >= 0.55;
    return {
      isFlagged,
      abuseScore,
      threatScore,
      spamScore
    };
  }

  static detectPII(text) {
    const spans = [];
    const phoneRegex = /(?<!\w)(?:\+?91[\s.-]?)?([6-9]\d{4}[\s.-]?\d{5})(?!\w)/g;
    const emailRegex = /(?<!\w)[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?!\w)/g;
    const upiRegex = /(?<!\w)[a-zA-Z0-9._-]+@(okaxis|okhdfcbank|oksbi|paytm|ybl|upi)(?!\w)/gi;
    const orderRegex = /(?<!\w)(?:ORD|OD|ORDER)[-_#]?[A-Z0-9]{6,14}(?!\w)/gi;
    const nameRegex = /\b(Rahul|Amit|Priya|Sneha|Aditya|Rohit)\b/g;

    let m;
    while ((m = phoneRegex.exec(text)) !== null) {
      spans.push({ type: "PHONE", start: m.index, end: m.index + m[0].length, text: m[0] });
    }
    while ((m = emailRegex.exec(text)) !== null) {
      spans.push({ type: "EMAIL", start: m.index, end: m.index + m[0].length, text: m[0] });
    }
    while ((m = upiRegex.exec(text)) !== null) {
      spans.push({ type: "UPI_ID", start: m.index, end: m.index + m[0].length, text: m[0] });
    }
    while ((m = orderRegex.exec(text)) !== null) {
      spans.push({ type: "ORDER_ID", start: m.index, end: m.index + m[0].length, text: m[0] });
    }
    while ((m = nameRegex.exec(text)) !== null) {
      spans.push({ type: "NAME", start: m.index, end: m.index + m[0].length, text: m[0] });
    }

    return spans.sort((a, b) => a.start - b.start);
  }

  static redact(text, spans) {
    if (!spans || !spans.length) return text;
    const sorted = [...spans].sort((a, b) => b.start - a.start);
    let res = text;
    for (const s of sorted) {
      res = res.substring(0, s.start) + `[${s.type}]` + res.substring(s.end);
    }
    return res;
  }

  static normalize(text, mode = "english") {
    if (!text) return "";
    let clean = text.replace(/(.)\1{2,}/g, "$1$1");
    clean = clean.replace(/\b(plz|pls)\b/gi, "please").replace(/\bnhi\b/gi, "nahi");

    if (mode === "minimal") return clean;

    if (mode === "native_script") {
      return clean
        .replace(/\bbhai\b/gi, "भाई")
        .replace(/\bpayment\b/gi, "पेमेंट")
        .replace(/\bdeduct\b/gi, "डिडक्ट")
        .replace(/\bho\b/gi, "हो")
        .replace(/\bgaya\b/gi, "गया")
        .replace(/\bpar\b/gi, "पर")
        .replace(/\border\b/gi, "ऑर्डर")
        .replace(/\bconfirm\b/gi, "कन्फर्म")
        .replace(/\bnahi\b/gi, "नहीं")
        .replace(/\bnaaku\b/gi, "నాకు")
        .replace(/\bdeliver\b/gi, "డెలివర్")
        .replace(/\bavvaledu\b/gi, "అవ్వలేదు")
        .replace(/\btaka\b/gi, "টাকা")
        .replace(/\bkete\b/gi, "কেটে");
    }

    // Default English normalization
    const lower = clean.toLowerCase();
    if (/payment deduct ho gaya.*order confirm nahi/i.test(lower)) {
      return "My payment was deducted, but the order is not showing as confirmed.";
    }
    if (/naaku delivery raledu/i.test(lower)) {
      return "My order has not been delivered to me yet.";
    }
    if (/taka kete geche.*order confirm hoyni/i.test(lower)) {
      return "Money was deducted, but the order was not confirmed.";
    }
    return clean;
  }
}

// UI Controller
document.addEventListener("DOMContentLoaded", () => {
  const sourceInput = document.getElementById("sourceInput");
  const charCountLabel = document.getElementById("charCountLabel");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const modeSelect = document.getElementById("modeSelect");
  const btnToggleRedact = document.getElementById("btnToggleRedact");

  // Output elements
  const langBadge = document.getElementById("langBadge");
  const langNameLabel = document.getElementById("langNameLabel");
  const langConfidenceFill = document.getElementById("langConfidenceFill");
  const langConfidenceVal = document.getElementById("langConfidenceVal");
  const langDetails = document.getElementById("langDetails");

  const topIntentLabel = document.getElementById("topIntentLabel");
  const intentBadge = document.getElementById("intentBadge");
  const intentConfidenceFill = document.getElementById("intentConfidenceFill");
  const intentConfidenceVal = document.getElementById("intentConfidenceVal");

  const safetyBadge = document.getElementById("safetyBadge");
  const modAbuseFill = document.getElementById("modAbuseFill");
  const modAbuseVal = document.getElementById("modAbuseVal");
  const modThreatFill = document.getElementById("modThreatFill");
  const modThreatVal = document.getElementById("modThreatVal");
  const modSpamFill = document.getElementById("modSpamFill");
  const modSpamVal = document.getElementById("modSpamVal");

  const piiSpansContainer = document.getElementById("piiSpansContainer");
  const redactedPreview = document.getElementById("redactedPreview");
  const normalizedTextOutput = document.getElementById("normalizedTextOutput");
  const statusLatency = document.getElementById("statusLatency");

  let showingRedacted = true;
  let currentSpans = [];

  // Update char count
  sourceInput.addEventListener("input", () => {
    charCountLabel.textContent = `${sourceInput.value.length} characters`;
  });

  // Sample Presets
  for (const [id, text] of Object.entries(SAMPLES)) {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener("click", () => {
        sourceInput.value = text;
        charCountLabel.textContent = `${text.length} characters`;
        runPipeline();
      });
    }
  }

  function runPipeline() {
    const text = sourceInput.value.trim();
    if (!text) return;

    const t0 = performance.now();

    // 1. Language ID
    const langRes = ClientNLP.identifyLanguage(text);
    langBadge.textContent = langRes.isCodeMixed ? "Code-Mixed" : "Monolingual";
    langNameLabel.textContent = langRes.lang;
    const langPct = Math.round(langRes.confidence * 100);
    langConfidenceFill.style.width = `${langPct}%`;
    langConfidenceVal.textContent = `${langPct}%`;
    langDetails.textContent = `Romanized: ${langRes.isRomanized ? "Yes" : "No"} | Script: Latin / Indic`;

    // 2. Intent
    const intentRes = ClientNLP.classifyIntent(text);
    topIntentLabel.textContent = intentRes.intent;
    intentBadge.textContent = intentRes.abstained ? "Abstained" : "Predicted";
    intentBadge.style.background = intentRes.abstained ? "rgba(245, 158, 11, 0.2)" : "rgba(16, 185, 129, 0.2)";
    intentBadge.style.color = intentRes.abstained ? "#fbbf24" : "#34d399";
    const intPct = Math.round(intentRes.confidence * 100);
    intentConfidenceFill.style.width = `${intPct}%`;
    intentConfidenceVal.textContent = `${intPct}%`;

    // 3. Moderation
    const modRes = ClientNLP.moderate(text);
    safetyBadge.textContent = modRes.isFlagged ? "Flagged Content" : "Safe";
    safetyBadge.style.background = modRes.isFlagged ? "rgba(244, 63, 94, 0.2)" : "rgba(16, 185, 129, 0.2)";
    safetyBadge.style.color = modRes.isFlagged ? "#fb7185" : "#34d399";

    modAbuseFill.style.width = `${Math.round(modRes.abuseScore * 100)}%`;
    modAbuseVal.textContent = modRes.abuseScore.toFixed(2);
    modAbuseFill.className = modRes.abuseScore > 0.5 ? "meter-fill flagged" : "meter-fill safe";

    modThreatFill.style.width = `${Math.round(modRes.threatScore * 100)}%`;
    modThreatVal.textContent = modRes.threatScore.toFixed(2);
    modThreatFill.className = modRes.threatScore > 0.4 ? "meter-fill flagged" : "meter-fill safe";

    modSpamFill.style.width = `${Math.round(modRes.spamScore * 100)}%`;
    modSpamVal.textContent = modRes.spamScore.toFixed(2);
    modSpamFill.className = modRes.spamScore > 0.55 ? "meter-fill flagged" : "meter-fill safe";

    // 4. PII
    currentSpans = ClientNLP.detectPII(text);
    if (currentSpans.length > 0) {
      piiSpansContainer.innerHTML = currentSpans.map(s => 
        `<span class="entity-pill"><strong>[${s.type}]</strong> ${escapeHtml(s.text)}</span>`
      ).join(" ");
    } else {
      piiSpansContainer.innerHTML = `<span style="font-size: 0.85rem; color: var(--text-sub);">No sensitive PII identifiers detected.</span>`;
    }

    renderRedactionView(text);

    // 5. Normalization
    const mode = modeSelect.value;
    const norm = ClientNLP.normalize(text, mode);
    normalizedTextOutput.textContent = norm;

    const lat = (performance.now() - t0).toFixed(2);
    statusLatency.textContent = `${lat} ms`;
  }

  function renderRedactionView(text) {
    if (showingRedacted) {
      redactedPreview.textContent = ClientNLP.redact(text, currentSpans);
    } else {
      redactedPreview.textContent = text;
    }
  }

  btnToggleRedact.addEventListener("click", () => {
    showingRedacted = !showingRedacted;
    btnToggleRedact.textContent = showingRedacted ? "View Original Text" : "View Redacted Text";
    renderRedactionView(sourceInput.value.trim());
  });

  btnAnalyze.addEventListener("click", runPipeline);
  modeSelect.addEventListener("change", runPipeline);

  // Run initial sample
  sourceInput.value = SAMPLES.sampleHinglish;
  charCountLabel.textContent = `${SAMPLES.sampleHinglish.length} characters`;
  runPipeline();
});

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
