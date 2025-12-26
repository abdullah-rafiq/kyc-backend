const express = require('express');
const cors = require('cors');
const { InferenceClient, HfInference } = require('@huggingface/inference');
const admin = require('firebase-admin');
const fetch = require('node-fetch').default;
const { spawn } = require('child_process');
const path = require('path');
require('dotenv').config();
const app = express();
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ limit: '20mb', extended: true }));
app.use(cors());
app.use(express.json());

const firebaseServiceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT;
const firebaseServiceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH;

if (!admin.apps.length) {
  if (firebaseServiceAccountJson && String(firebaseServiceAccountJson).trim().length > 0) {
    admin.initializeApp({
      credential: admin.credential.cert(JSON.parse(firebaseServiceAccountJson)),
    });
  } else if (firebaseServiceAccountPath && String(firebaseServiceAccountPath).trim().length > 0) {
    // eslint-disable-next-line global-require, import/no-dynamic-require
    const serviceAccount = require(path.resolve(__dirname, firebaseServiceAccountPath));
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
    });
  } else {
    admin.initializeApp();
  }
}

// Initialize Firestore
const db = admin.firestore();

// Authentication Middleware
const authMiddleware = async (req, res, next) => {
  const header = req.headers.authorization;
  if (!header || !header.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized: No token provided' });
  }
  const token = header.split('Bearer ')[1];
  try {
    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded;
    next();
  } catch (err) {
    console.error('Auth error:', err);
    return res.status(401).json({ error: 'Unauthorized: Invalid token' });
  }
};

app.get('/__version', (req, res) => {
  res.json({
    service: 'ai-backedn',
    mode: 'kyc_bridge',
    time: new Date().toISOString(),
    env: {
      hasKycApiUrl: Boolean(process.env.KYC_API_URL && String(process.env.KYC_API_URL).trim().length > 0),
      kycApiUrl: process.env.KYC_API_URL || null,
      hasHfApiKey: Boolean(process.env.HUGGINGFACE_API_KEY && String(process.env.HUGGINGFACE_API_KEY).trim().length > 0),
      hfChatModel: process.env.HF_CHAT_MODEL || null,
      firebaseServiceAccountPath: process.env.FIREBASE_SERVICE_ACCOUNT_PATH || null,
      hasFirebaseServiceAccountJson: Boolean(
        process.env.FIREBASE_SERVICE_ACCOUNT && String(process.env.FIREBASE_SERVICE_ACCOUNT).trim().length > 0,
      ),
    },
    render: {
      serviceId: process.env.RENDER_SERVICE_ID || null,
      gitCommit: process.env.RENDER_GIT_COMMIT || null,
    },
  });
});

app.get('/__auth_check', authMiddleware, (req, res) => {
  res.json({
    ok: true,
    uid: req.user?.uid || null,
    email: req.user?.email || null,
    issuer: req.user?.iss || null,
    audience: req.user?.aud || null,
  });
});

// Python Bridge: runKycEngine
async function runKycEngine(mode, data) {
  const apiUrl = process.env.KYC_API_URL;
  if (apiUrl && String(apiUrl).trim().length > 0) {
    let endpoint = null;
    if (mode === 'cnic') endpoint = '/verify-cnic';
    if (mode === 'face') endpoint = '/face-verify';
    if (mode === 'liveness') endpoint = '/face-liveness';
    if (mode === 'shop') endpoint = '/shop-verify';
    if (!endpoint) throw new Error(`Unknown KYC mode: ${mode}`);

    const base = String(apiUrl).replace(/\/$/, '');
    const url = `${base}${endpoint}`;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data || {}),
      });

      const text = await resp.text();
      let parsed;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = { raw: text };
      }

      if (resp.ok) {
        return parsed;
      }

      if (resp.status >= 500 && resp.status <= 599 && attempt === 0) {
        await new Promise((r) => setTimeout(r, 900));
        continue;
      }

      throw new Error(`KYC engine error (${resp.status}) for ${url}: ${text}`);
    }
  }

  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', ['hf_kyc_space/app.py', mode]);

    let outputData = '';
    let errorData = '';

    const inputJSON = JSON.stringify(data);

    pythonProcess.stdout.on('data', (chunk) => {
      outputData += chunk.toString();
    });

    pythonProcess.stderr.on('data', (chunk) => {
      errorData += chunk.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`KYC Engine failed (code ${code}): ${errorData}`));
      }
      try {
        const lines = outputData.trim().split('\n');
        const jsonLine = lines[lines.length - 1];
        const result = JSON.parse(jsonLine);
        resolve(result);
      } catch (e) {
        reject(new Error(`Failed to parse KYC Engine output: ${e.message}. Raw output: ${outputData}`));
      }
    });

    pythonProcess.stdin.write(inputJSON);
    pythonProcess.stdin.end();
  });
}

// -------------------- KYC Helpers --------------------

function confidenceScore(value, opts = {}) {
  if (!value) return 0.0;

  let score = 0.4; // base OCR presence

  if (opts.formatOk) score += 0.3;
  if (opts.fromLLM) score += 0.2;
  if (opts.expectedMatch) score += 0.1;

  return Math.min(1.0, Number(score.toFixed(2)));
}

function normalizeUrduAddress(text) {
  if (!text) return null;

  return text
    .replace(/[^\u0600-\u06FF\s،]/g, '')
    .replace(/\s+/g, ' ')
    .replace(/،+/g, '،')
    .trim();
}

function isLowQuality(confidence) {
  const critical = ['fullName', 'cnicNumber', 'dateOfBirth'];
  return critical.some(k => confidence[k] < 0.5);
}

function normalizeGender(g) {
  if (!g) return null;
  const s = g.toLowerCase();
  if (s.includes('male') || s.includes('man')) return 'male';
  if (s.includes('female') || s.includes('woman')) return 'female';
  return null;
}

function inferGenderFromCnicName(name) {
  if (!name) return null;
  const femaleHints = ['bibi', 'begum', 'fatima', 'aisha', 'zainab'];
  const lower = name.toLowerCase();
  if (femaleHints.some(h => lower.includes(h))) return 'female';
  return 'male'; // default for PK CNIC
}

function faceCnicMatch({ faceResult, cnic }) {
  const faceGender = normalizeGender(faceResult?.gender);
  const cnicGender = inferGenderFromCnicName(cnic?.fullName);

  const distance = typeof faceResult?.distance === 'number'
    ? faceResult.distance
    : (typeof faceResult?.similarity === 'number' ? faceResult.similarity : null);
  const threshold = typeof faceResult?.threshold === 'number' ? faceResult.threshold : null;
  const similarityOk = faceResult?.verified === true
    ? true
    : (distance != null && threshold != null)
      ? distance <= threshold
      : (distance != null ? distance < 0.35 : null);

  return {
    faceVerified: faceResult?.verified === true,
    genderMatch: faceGender && cnicGender
      ? faceGender === cnicGender
      : null,
    similarityOk,
  };
}

function parseDateDMY(d) {
  if (!d) return null;
  const [dd, mm, yyyy] = d.split(/[\/.-]/).map(Number);
  if (!dd || !mm || !yyyy) return null;
  return new Date(yyyy, mm - 1, dd);
}

function isCnicExpired(expiryDate) {
  const exp = parseDateDMY(expiryDate);
  if (!exp) return null;
  return exp < new Date();
}

function antiSpoofingHeuristics(faceResult) {
  let score = 0;

  if (faceResult?.verified === true) score += 0.6;
  if (faceResult?.confidence > 0.85) score += 0.25;
  if (faceResult?.sharpness > 0.6) score += 0.1;
  if (faceResult?.reflectionDetected === false) score += 0.03;
  if (faceResult?.multipleFaces === false) score += 0.02;

  return Number(Math.min(1.0, score).toFixed(2));
}

function findLabeledValue(keywords, lines) {
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lower = line.toLowerCase();

    if (keywords.some(k => lower.includes(k))) {
      const parts = line.split(':');
      if (parts.length > 1 && parts[1].trim()) {
        return parts.slice(1).join(':').trim();
      }

      const next = lines[i + 1];
      if (next && !keywords.some(k => next.toLowerCase().includes(k))) {
        return next.trim();
      }
    }
  }
  return null;
}

function isReplyInLanguage(text, language) {
  const t = (text || '').trim();
  const lang = (language || '').toLowerCase();

  if (!t) return true;

  const hasUrduScript = /[\u0600-\u06FF]/.test(t);

  if (lang === 'urdu') {
    // Must contain Urdu/Arabic script
    return hasUrduScript;
  }

  if (lang === 'english' || lang === 'roman_urdu') {
    // Must NOT contain Urdu/Arabic script
    return !hasUrduScript;
  }

  // Unknown language -> don't block
  return true;
}

function normalizeBase64(input) {
  if (input == null) return input;
  const s = String(input);
  const commaIndex = s.indexOf(',');
  const clean = commaIndex >= 0 ? s.slice(commaIndex + 1) : s;
  return clean.replace(/\s/g, '');
}

function hasUrduScript(text) {
  return /[\u0600-\u06FF]/.test(text || '');
}

function formatCnicNumber(raw) {
  if (raw == null) return null;
  const digits = String(raw).replace(/\D/g, '');
  if (digits.length !== 13) return null;
  return `${digits.slice(0, 5)}-${digits.slice(5, 12)}-${digits.slice(12)}`;
}

function isMissingCnicFields(extracted) {
  if (!extracted || typeof extracted !== 'object') return true;
  const fullName = String(extracted.fullName || '').trim();
  const cnicNumber = String(extracted.cnicNumber || '').trim();
  const dateOfBirth = String(extracted.dateOfBirth || '').trim();
  return !(fullName && cnicNumber && dateOfBirth);
}

function extractFirstJsonObject(text) {
  const s = String(text || '').trim();
  const start = s.indexOf('{');
  const end = s.lastIndexOf('}');
  if (start >= 0 && end > start) return s.slice(start, end + 1);
  return s;
}

async function callChatModel(systemPrompt, userContent) {
  const token = process.env.HUGGINGFACE_API_KEY;
  const model = process.env.HF_CHAT_MODEL;
  if (!token || String(token).trim().length === 0) {
    throw new Error('HUGGINGFACE_API_KEY is not configured');
  }
  if (!model || String(model).trim().length === 0) {
    throw new Error('HF_CHAT_MODEL is not configured');
  }

  const hf = new InferenceClient(token);
  const out = await hf.chatCompletion({
    model,
    messages: [
      { role: 'system', content: String(systemPrompt || '') },
      { role: 'user', content: String(userContent || '') },
    ],
    max_tokens: 600,
    temperature: 0.1,
  });

  const content = out?.choices?.[0]?.message?.content;
  if (!content || String(content).trim().length === 0) {
    throw new Error('Chat model returned empty content');
  }
  return String(content);
}

app.post('/ai/support/ask', authMiddleware, async (req, res) => {
  try {
    const { message, language } = req.body || {};
    const text = String(message || '').trim();
    if (!text) {
      return res.status(400).json({ error: 'message is required' });
    }

    const lang = String(language || '').trim();
    const systemPrompt = `You are the customer support assistant for the Assist app.
Reply clearly and briefly.
If you cannot help, ask the user to contact support@assist.com.
Reply in ${lang || 'English'}.`;

    const reply = await callChatModel(systemPrompt, text);
    return res.json({ reply });
  } catch (err) {
    console.error('Support ask error:', err);
    return res.status(500).json({
      error: 'Support request failed',
      details: err?.message ?? String(err),
    });
  }
});

async function extractCnicWithChatModel({ frontLines = [], backLines = [], expectedName, expectedFatherName, expectedDob }) {
  const systemPrompt = `You are a data extraction engine for Pakistan CNIC OCR results.
Return ONLY valid JSON.
If a field is not present, use null.

Output schema:
{
  "fullName": string|null,
  "fatherName": string|null,
  "cnicNumber": string|null,
  "dateOfBirth": string|null,
  "dateOfIssue": string|null,
  "dateOfExpiry": string|null,
  "addressUrdu": {"line1": string|null} | null
}

Important:
- CNIC number format should be #####-#######-# when possible.
- Dates should be DD/MM/YYYY when possible.
- OCR text may include Urdu and English.
`.trim();

  const payload = {
    frontLines,
    backLines,
    expected: {
      name: expectedName || null,
      fatherName: expectedFatherName || null,
      dob: expectedDob || null,
    },
  };

  const raw = await callChatModel(systemPrompt, JSON.stringify(payload));
  const parsed = JSON.parse(extractFirstJsonObject(raw));
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('CNIC extraction model returned invalid JSON');
  }

  const normalized = {
    fullName: parsed.fullName ?? null,
    fatherName: parsed.fatherName ?? null,
    cnicNumber: parsed.cnicNumber ?? null,
    dateOfBirth: parsed.dateOfBirth ?? null,
    dateOfIssue: parsed.dateOfIssue ?? null,
    dateOfExpiry: parsed.dateOfExpiry ?? null,
    addressUrdu: parsed.addressUrdu ?? null,
  };

  const formatted = formatCnicNumber(normalized.cnicNumber) || normalized.cnicNumber;
  normalized.cnicNumber = formatted;

  return normalized;
}

function extractCnicInfo({ frontLines = [], backLines = [] }) {
  const allLines = [...frontLines, ...backLines]
    .map(l => String(l || '').trim())
    .filter(Boolean);

  const joined = allLines.join(' ');

  const cnicMatch =
    joined.match(/\d{5}-\d{7}-\d/) ||
    joined.match(/\d{13}/);

  const cnicNumber = cnicMatch
    ? formatCnicNumber(cnicMatch[0])
    : null;

  const dates = joined.match(/\d{2}[\/.-]\d{2}[\/.-]\d{4}/g) || [];

  const dateOfBirth = dates[0] || null;
  const dateOfIssue = dates[1] || null;
  const dateOfExpiry = dates[2] || null;

  const fullName = findLabeledValue(['name'], frontLines);
  const fatherName = findLabeledValue(['father', 'husband'], frontLines);

  const urduLines = backLines.filter(hasUrduScript);
  const addressUrdu =
    urduLines.length > 0
      ? { line1: urduLines.join('، ') }
      : null;

  return {
    fullName,
    fatherName,
    cnicNumber,
    dateOfBirth,
    dateOfIssue,
    dateOfExpiry,
    addressUrdu,
  };
}

app.post('/api/vision/verify-cnic', authMiddleware, async (req, res) => {
  try {
    const { cnicFrontBase64, cnicBackBase64, expectedName, expectedFatherName, expectedDob } = req.body;
    if (!cnicFrontBase64) {
      return res.status(400).json({ error: 'cnicFrontBase64 is required' });
    }

    console.log("Processing CNIC via Local Engine...");
    const frontResult = await runKycEngine('cnic', {
      image: normalizeBase64(cnicFrontBase64),
    });

    let backResult = null;
    let backError = null;
    if (cnicBackBase64) {
      try {
        backResult = await runKycEngine('cnic', {
          image: normalizeBase64(cnicBackBase64),
        });
      } catch (e) {
        backError = e?.message ?? String(e);
        console.warn('CNIC back OCR failed (continuing with front only):', backError);
      }
    }

    const frontLines = Array.isArray(frontResult?.raw_text)
      ? frontResult.raw_text
      : (Array.isArray(frontResult?.rawText) ? frontResult.rawText : []);
    const backLines = Array.isArray(backResult?.raw_text)
      ? backResult.raw_text
      : (Array.isArray(backResult?.rawText) ? backResult.rawText : []);

    const extracted = extractCnicInfo({
      frontLines,
      backLines,
    });

    const providerExtracted = {
      fullName: frontResult?.fullName ?? null,
      fatherName: frontResult?.fatherName ?? null,
      cnicNumber: frontResult?.cnicNumber ?? null,
      dateOfBirth: frontResult?.dateOfBirth ?? null,
      dateOfIssue: frontResult?.dateOfIssue ?? null,
      dateOfExpiry: frontResult?.dateOfExpiry ?? null,
      addressUrdu: backResult?.addressUrdu ?? frontResult?.addressUrdu ?? null,
    };

    for (const key of [
      'fullName',
      'fatherName',
      'cnicNumber',
      'dateOfBirth',
      'dateOfIssue',
      'dateOfExpiry',
      'addressUrdu',
    ]) {
      if (extracted[key] == null || String(extracted[key] || '').trim().length === 0) {
        extracted[key] = providerExtracted[key];
      }
    }

    if (extracted.cnicNumber) {
      extracted.cnicNumber = formatCnicNumber(extracted.cnicNumber) || extracted.cnicNumber;
    }

    if (isMissingCnicFields(extracted) && frontLines.length > 0) {
      try {
        const llm = await extractCnicWithChatModel({
          frontLines,
          backLines,
          expectedName,
          expectedFatherName,
          expectedDob,
        });

        for (const key of [
          'fullName',
          'fatherName',
          'cnicNumber',
          'dateOfBirth',
          'dateOfIssue',
          'dateOfExpiry',
          'addressUrdu',
        ]) {
          if (extracted[key] == null || String(extracted[key] || '').trim().length === 0) {
            extracted[key] = llm[key];
          }
        }
      } catch (e) {
        console.warn('CNIC LLM extraction failed:', e?.message ?? String(e));
      }
    }

    // ... (extraction logic previously executed)

    // Calculate Confidence Scores
    const confidence = {
      fullName: confidenceScore(extracted.fullName, {
        expectedMatch: expectedName &&
          extracted.fullName?.toLowerCase().includes(expectedName.toLowerCase()),
      }),

      fatherName: confidenceScore(extracted.fatherName),

      cnicNumber: confidenceScore(extracted.cnicNumber, {
        formatOk: /^\d{5}-\d{7}-\d$/.test(extracted.cnicNumber || ''),
      }),

      dateOfBirth: confidenceScore(extracted.dateOfBirth),
      dateOfIssue: confidenceScore(extracted.dateOfIssue),
      dateOfExpiry: confidenceScore(extracted.dateOfExpiry),

      addressUrdu: confidenceScore(extracted.addressUrdu?.line1),
    };

    // Normalize Urdu Address
    if (extracted.addressUrdu && extracted.addressUrdu.line1) {
      extracted.addressUrdu.line1 = normalizeUrduAddress(extracted.addressUrdu.line1);
    }

    // Quality & Expiry Checks
    let status = 'auto_verified';
    if (isLowQuality(confidence)) {
      status = 'needs_manual_review';
    }

    const expired = isCnicExpired(extracted.dateOfExpiry);
    if (expired === true) {
      status = 'rejected_expired_cnic';
    }

    const responsePayload = {
      extracted,
      confidence,
      status, // Return status to frontend
      raw: {
        front: frontLines,
        back: backLines,
        backError,
      },
    };

    // Save to DB logic
    if (req.user && req.user.uid) {
      try {
        const userRef = db.collection('users').doc(req.user.uid);
        await userRef.set({
          verification: {
            cnic: {
              extracted,
              confidence,
              status,
              expired, // explicit flag
              raw: responsePayload.raw,
              updatedAt: admin.firestore.FieldValue.serverTimestamp(),
            }
          }
        }, { merge: true });
      } catch (e) { console.error("Error saving CNIC data:", e); }
    }

    return res.json(responsePayload);
  } catch (err) {
    console.error('CNIC error:', err);
    return res.status(500).json({ error: 'CNIC verification failed', details: err?.message ?? String(err) });
  }
});

// 2. Face Verification
app.post('/api/kyc/face', authMiddleware, async (req, res) => {
  try {
    const { cnicImage, selfieImage } = req.body;
    if (!cnicImage || !selfieImage) {
      return res.status(400).json({ error: 'cnicImage and selfieImage required' });
    }

    console.log("Verifying Face via Local Engine...");
    const faceResult = await runKycEngine('face', { image: normalizeBase64(cnicImage), image2: normalizeBase64(selfieImage) });

    // Fetch previously extracted CNIC data for cross-matching
    let cnicData = null;
    if (req.user && req.user.uid) {
      const userDoc = await db.collection('users').doc(req.user.uid).get();
      cnicData = userDoc.data()?.verification?.cnic?.extracted;
    }

    // Match Face <-> CNIC
    const matchResult = faceCnicMatch({ faceResult, cnic: cnicData });

    // Anti-Spoofing
    let livenessResult = null;
    try {
      livenessResult = await runKycEngine('liveness', { image: normalizeBase64(selfieImage) });
    } catch (e) {
      console.warn('Liveness check failed (fallback to heuristics):', e?.message ?? String(e));
    }

    const livenessScore = typeof livenessResult?.livenessScore === 'number'
      ? livenessResult.livenessScore
      : (typeof livenessResult?.spoofScore === 'number' ? livenessResult.spoofScore : null);

    const spoofScore = typeof livenessScore === 'number'
      ? livenessScore
      : antiSpoofingHeuristics(faceResult);
    let status = 'auto_verified';

    const minLiveness = Number.parseFloat(process.env.FACE_LIVENESS_MIN || '0.6');
    const livenessMin = Number.isFinite(minLiveness) ? minLiveness : 0.6;

    if (spoofScore < livenessMin) {
      status = 'needs_manual_review_spoof_risk';
    }
    if (matchResult.similarityOk === false) {
      status = 'rejected_face_mismatch';
    }

    const finalResult = {
      ...faceResult,
      match: matchResult,
      spoofScore,
      status
    };

    // Save to Firestore
    if (req.user && req.user.uid) {
      await db.collection('users').doc(req.user.uid).set({
        verification: {
          face: {
            status,
            verified: faceResult.verified,
            similarity: faceResult.distance,
            distance: faceResult.distance,
            threshold: faceResult.threshold,
            confidence: faceResult.confidence,
            gender: faceResult.gender,
            spoofScore,
            livenessScore: typeof livenessScore === 'number' ? livenessScore : null,
            isReal: typeof livenessResult?.isReal === 'boolean' ? livenessResult.isReal : null,
            spoofModel: livenessResult?.spoofModel ?? null,
            match: matchResult,
            updatedAt: admin.firestore.FieldValue.serverTimestamp(),
          },
          status: status === 'auto_verified' ? 'auto_verified' : status // Update overall status strictly if verified
        }
      }, { merge: true });
    }

    return res.json(finalResult);
  } catch (err) {
    console.error('Face verification error:', err);
    return res.status(500).json({
      error: 'Face verification failed',
      details: err?.message ?? String(err),
    });
  }
});

// 3. Shop/Tool Verification
app.post('/api/kyc/shop', authMiddleware, async (req, res) => {
  try {
    const { shopImage } = req.body;
    if (!shopImage) {
      return res.status(400).json({ error: 'shopImage is required' });
    }

    console.log("Verifying Shop via Local Engine...");
    const result = await runKycEngine('shop', { image: normalizeBase64(shopImage) });

    const status = typeof result?.status === 'string'
      ? result.status
      : (typeof result?.score === 'number'
        ? (result.score >= 0.6 ? 'auto_verified' : 'needs_manual_review')
        : null);

    if (req.user && req.user.uid) {
      await db.collection('users').doc(req.user.uid).set({
        verification: {
          shop: {
            status,
            score: typeof result?.score === 'number' ? result.score : null,
            shape: result?.shape ?? null,
            detected_objects: result?.detected_objects ?? result?.detectedObjects ?? null,
            tools: result?.tools ?? null,
            text_content: result?.text_content ?? result?.textContent ?? result?.ocrText ?? null,
            notes: result?.notes ?? null,
            topService: result?.topService ?? null,
            serviceCandidates: result?.serviceCandidates ?? null,
            serviceHints: result?.serviceHints ?? null,
            updatedAt: admin.firestore.FieldValue.serverTimestamp(),
          },
        }
      }, { merge: true });
    }

    return res.json(result);
  } catch (err) {
    console.error('Shop verification error:', err);
    return res.status(500).json({ error: 'Shop verification failed', details: err?.message ?? String(err) });
  }
});

// -------------------- Admin KYC Routes --------------------

app.get('/admin/kyc/pending', authMiddleware, async (req, res) => {
  // Ideally check if req.user is admin
  try {
    const snap = await db.collection('users')
      .where('verification.status', '==', 'needs_manual_review')
      .limit(50)
      .get();

    const list = snap.docs.map(d => ({
      uid: d.id,
      ...d.data().verification,
    }));

    res.json(list);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/admin/kyc/decision', authMiddleware, async (req, res) => {
  try {
    const { uid, decision, reason } = req.body;

    await db.collection('users').doc(uid).update({
      'verification.status': decision,
      'verification.reviewReason': reason || null,
      'verification.reviewedAt': admin.firestore.FieldValue.serverTimestamp(),
    });

    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// -------------------- Start Server --------------------

const PORT = process.env.PORT || 8080;
app.listen(PORT, () =>
  console.log(`AI backend listening on port ${PORT}`),
);