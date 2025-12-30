const express = require('express');
const cors = require('cors');
const { InferenceClient, HfInference } = require('@huggingface/inference');
const admin = require('firebase-admin');
const crypto = require('crypto');
const querystring = require('querystring');
const { spawn } = require('child_process');
const path = require('path');
require('dotenv').config();
const app = express();
app.set('trust proxy', 1);
app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ limit: '30mb', extended: true }));
app.use(cors());

app.use((err, req, res, next) => {
  if (err && (err.type === 'entity.too.large' || err.status === 413)) {
    return res.status(413).json({
      error: 'Payload too large. Reduce image size or increase server body limit.',
      errorCode: 'PAYLOAD_TOO_LARGE',
    });
  }
  return next(err);
});

const fetch = globalThis.fetch;
if (typeof fetch !== 'function') {
  throw new Error('Global fetch is not available. Please run this service on Node.js >= 18.');
}

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

async function getUserRole(uid) {
  if (!uid) return null;
  try {
    const snap = await db.collection('users').doc(String(uid)).get();
    if (!snap.exists) return null;
    const role = snap.get('role');
    return typeof role === 'string' ? role : null;
  } catch (_) {
    return null;
  }
}

app.post('/provider/stats/recompute', authMiddleware, async (req, res) => {
  try {
    const requesterUid = req.user?.uid;
    if (!requesterUid) {
      return res.status(401).json({ error: 'Unauthorized: Invalid token' });
    }

    const rawProviderId = req.body?.providerId;
    const providerId = String(rawProviderId || requesterUid).trim();
    if (!providerId) {
      return res.status(400).json({ error: 'providerId is required' });
    }

    if (providerId !== requesterUid) {
      const role = await getUserRole(requesterUid);
      if (role !== 'admin') {
        let hasBooking = false;
        try {
          const bookingSnap = await db
            .collection('bookings')
            .where('customerId', '==', requesterUid)
            .where('providerId', '==', providerId)
            .limit(1)
            .get();
          hasBooking = !bookingSnap.empty;
        } catch (_) {
          // Avoid composite index requirements by falling back to filtering in code.
          const bookingSnap = await db
            .collection('bookings')
            .where('customerId', '==', requesterUid)
            .limit(50)
            .get();
          hasBooking = bookingSnap.docs.some((d) => String(d.get('providerId') || '') === providerId);
        }

        if (!hasBooking) {
          return res.status(403).json({ error: 'Forbidden' });
        }
      }
    }

    const servicesSnap = await db
      .collection('services')
      .where('providerId', '==', providerId)
      .get();

    let minActive = null;
    let minAny = null;
    for (const doc of servicesSnap.docs) {
      const data = doc.data() || {};
      const p = Number(data.basePrice || 0);
      if (!Number.isFinite(p) || p <= 0) continue;
      if (minAny == null || p < minAny) minAny = p;
      if (data.isActive !== false) {
        if (minActive == null || p < minActive) minActive = p;
      }
    }
    const minPrice = minActive ?? minAny;

    const reviewsSnap = await db
      .collection('reviews')
      .where('providerId', '==', providerId)
      .get();

    let ratingTotal = 0;
    let ratingCount = 0;
    for (const doc of reviewsSnap.docs) {
      const data = doc.data() || {};
      const r = Number(data.rating || 0);
      if (!Number.isFinite(r) || r <= 0) continue;
      ratingTotal += r;
      ratingCount += 1;
    }
    const avgRating = ratingCount > 0 ? ratingTotal / ratingCount : 0;

    const bookingsSnap = await db
      .collection('bookings')
      .where('providerId', '==', providerId)
      .get();

    let completedJobs = 0;
    for (const doc of bookingsSnap.docs) {
      const data = doc.data() || {};
      if (String(data.status || '') === 'Completed') {
        completedJobs += 1;
      }
    }

    const publicStats = {
      minPrice: minPrice == null ? null : minPrice,
      avgRating,
      ratingCount,
      completedJobs,
      updatedAt: admin.firestore.FieldValue.serverTimestamp(),
    };

    await db.collection('users').doc(providerId).set(
      { publicStats },
      { merge: true },
    );

    return res.json({ ok: true, providerId, publicStats });
  } catch (err) {
    console.error('Provider stats recompute error:', err);
    return res.status(500).json({
      error: 'Provider stats recompute failed',
      details: err?.message ?? String(err),
    });
  }
});

function payfastMd5(input) {
  return crypto.createHash('md5').update(String(input), 'utf8').digest('hex');
}

function payfastSignature(params, passphrase) {
  const pairs = [];
  const keys = Object.keys(params || {})
    .filter((k) => k !== 'signature')
    .sort();

  for (const k of keys) {
    const v = params[k];
    if (v == null) continue;
    const s = String(v);
    if (s.trim().length === 0) continue;
    pairs.push(`${querystring.escape(k)}=${querystring.escape(s)}`);
  }

  if (passphrase && String(passphrase).trim().length > 0) {
    pairs.push(`passphrase=${querystring.escape(String(passphrase))}`);
  }

  return payfastMd5(pairs.join('&'));
}

function payfastProcessUrl() {
  const env = String(process.env.PAYFAST_ENV || '').trim().toLowerCase();
  if (env === 'sandbox') {
    return 'https://sandbox.payfast.co.za/eng/process';
  }
  return 'https://www.payfast.co.za/eng/process';
}

function baseUrlFromReq(req) {
  const proto = req.headers['x-forwarded-proto'] || req.protocol;
  const host = req.headers['x-forwarded-host'] || req.get('host');
  return `${proto}://${host}`;
}

function normalizeRemoteUrl(input) {
  if (input == null) return null;
  const s = String(input);
  const trimmed = s.trim();
  if (!trimmed) return null;
  return trimmed.replace(/\s+/g, '');
}

function assertAllowedRemoteUrl(urlString) {
  const uri = new URL(urlString);

  if (uri.protocol !== 'https:') {
    throw new Error(`Only https URLs are allowed: ${urlString}`);
  }

  const host = String(uri.hostname || '').toLowerCase();
  const allowed = host === 'res.cloudinary.com' || host.endsWith('.cloudinary.com');
  if (!allowed) {
    throw new Error(`Remote URL host not allowed: ${host}`);
  }
}

async function downloadRemoteImageAsBase64(urlString, label) {
  const normalized = normalizeRemoteUrl(urlString);
  if (!normalized) {
    throw new Error(`Missing ${label} URL`);
  }

  assertAllowedRemoteUrl(normalized);

  const controller = new AbortController();
  const timeoutMs = Number.parseInt(process.env.HTTP_DOWNLOAD_TIMEOUT_MS || '45000', 10);
  const timer = setTimeout(() => controller.abort(), Number.isFinite(timeoutMs) ? timeoutMs : 45000);

  try {
    const resp = await fetch(normalized, {
      method: 'GET',
      signal: controller.signal,
      redirect: 'follow',
    });

    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(`HTTP ${resp.status} while downloading ${label}: ${body}`);
    }

    const arrayBuffer = await resp.arrayBuffer();
    const buf = Buffer.from(arrayBuffer);
    return buf.toString('base64');
  } catch (e) {
    if (String(e?.name || '').toLowerCase().includes('abort')) {
      throw new Error(`Timed out downloading ${label} from ${normalized}`);
    }
    throw new Error(`Could not download ${label} from ${normalized}: ${e?.message ?? String(e)}`);
  } finally {
    clearTimeout(timer);
  }
}

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

app.get('/_version', (req, res) => {
  res.redirect(302, '/__version');
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

app.post('/api/payments/payfast/checkout', authMiddleware, async (req, res) => {
  try {
    const type = String(req.body?.type || '').trim();
    const bookingId = String(req.body?.bookingId || '').trim();
    const amountRaw = req.body?.amount;

    if (type !== 'booking' && type !== 'wallet_topup') {
      return res.status(400).json({ error: 'Invalid type' });
    }

    if (type === 'booking' && !bookingId) {
      return res.status(400).json({ error: 'bookingId is required' });
    }

    const uid = req.user?.uid;
    if (!uid) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    let amount = null;
    let itemName = null;

    if (type === 'booking') {
      const bookingSnap = await db.collection('bookings').doc(bookingId).get();
      if (!bookingSnap.exists) {
        return res.status(404).json({ error: 'Booking not found' });
      }
      const b = bookingSnap.data() || {};
      const price = typeof b.price === 'number' ? b.price : Number(b.price || 0);
      amount = Number.isFinite(price) ? price : 0;
      itemName = `Booking ${bookingId}`;
    } else {
      const parsed = typeof amountRaw === 'number' ? amountRaw : Number(amountRaw);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        return res.status(400).json({ error: 'amount is required' });
      }
      amount = parsed;
      itemName = 'Wallet top up';
    }

    if (!Number.isFinite(amount) || amount <= 0) {
      return res.status(400).json({ error: 'Invalid amount' });
    }

    const paymentRef = db.collection('payfast_payments').doc();
    await paymentRef.set({
      uid,
      type,
      bookingId: bookingId || null,
      amount,
      status: 'initiated',
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
    });

    const base = baseUrlFromReq(req);
    const url = `${base}/payfast/redirect/${paymentRef.id}`;
    return res.json({ url });
  } catch (err) {
    console.error('PayFast checkout error:', err);
    return res.status(500).json({
      error: 'Could not create PayFast checkout',
      details: err?.message ?? String(err),
    });
  }
});

app.get('/payfast/redirect/:paymentId', async (req, res) => {
  try {
    const paymentId = String(req.params.paymentId || '').trim();
    if (!paymentId) {
      return res.status(400).send('Missing paymentId');
    }

    const snap = await db.collection('payfast_payments').doc(paymentId).get();
    if (!snap.exists) {
      return res.status(404).send('Payment not found');
    }

    const data = snap.data() || {};

    const merchantId = String(process.env.PAYFAST_MERCHANT_ID || '').trim();
    const merchantKey = String(process.env.PAYFAST_MERCHANT_KEY || '').trim();
    const passphrase = String(process.env.PAYFAST_PASSPHRASE || '').trim();

    if (!merchantId || !merchantKey) {
      return res.status(500).send('PayFast is not configured');
    }

    const base = baseUrlFromReq(req);
    const notifyUrl = String(process.env.PAYFAST_NOTIFY_URL || '').trim() || `${base}/api/payments/payfast/itn`;
    const returnUrl = String(process.env.PAYFAST_RETURN_URL || '').trim() || `${base}/payfast/return`;
    const cancelUrl = String(process.env.PAYFAST_CANCEL_URL || '').trim() || `${base}/payfast/cancel`;

    const amount = Number(data.amount || 0);
    const itemName = data.type === 'booking'
      ? `Booking ${data.bookingId || ''}`
      : 'Wallet top up';

    const params = {
      merchant_id: merchantId,
      merchant_key: merchantKey,
      return_url: returnUrl,
      cancel_url: cancelUrl,
      notify_url: notifyUrl,
      m_payment_id: paymentId,
      amount: amount.toFixed(2),
      item_name: itemName,
      custom_str1: String(data.uid || ''),
      custom_str2: String(data.bookingId || ''),
    };

    const signature = payfastSignature(params, passphrase);
    params.signature = signature;

    const action = payfastProcessUrl();

    const inputs = Object.keys(params)
      .map((k) => {
        const v = String(params[k] ?? '');
        const safeKey = String(k)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
        const safeVal = v
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
        return `<input type="hidden" name="${safeKey}" value="${safeVal}" />`;
      })
      .join('');

    const html = `<!doctype html><html><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /><title>Redirecting…</title></head><body><form id="pf" method="post" action="${action}">${inputs}</form><script>document.getElementById('pf').submit();</script></body></html>`;
    res.set('Content-Type', 'text/html');
    return res.status(200).send(html);
  } catch (err) {
    console.error('PayFast redirect error:', err);
    return res.status(500).send('Could not redirect to PayFast');
  }
});

app.get('/payfast/return', async (req, res) => {
  return res.status(200).send('Payment submitted. You may return to the app and refresh.');
});

app.get('/payfast/cancel', async (req, res) => {
  return res.status(200).send('Payment cancelled. You may return to the app.');
});

app.post('/api/payments/payfast/itn', async (req, res) => {
  try {
    const passphrase = String(process.env.PAYFAST_PASSPHRASE || '').trim();
    const payload = req.body || {};
    const receivedSignature = String(payload.signature || '').trim();
    const paymentId = String(payload.m_payment_id || '').trim();

    if (!paymentId) {
      return res.status(400).send('Missing m_payment_id');
    }

    const expectedSignature = payfastSignature(payload, passphrase);
    if (!receivedSignature || receivedSignature !== expectedSignature) {
      return res.status(400).send('Invalid signature');
    }

    const paymentRef = db.collection('payfast_payments').doc(paymentId);
    const snap = await paymentRef.get();
    if (!snap.exists) {
      return res.status(404).send('Payment not found');
    }

    const payment = snap.data() || {};
    const status = String(payload.payment_status || '').trim();

    await paymentRef.set(
      {
        status: status || payment.status || 'unknown',
        payfast: payload,
        updatedAt: admin.firestore.FieldValue.serverTimestamp(),
      },
      { merge: true },
    );

    if (status === 'COMPLETE') {
      const amountGrossRaw = payload.amount_gross ?? payload.amount;
      const amountGross = typeof amountGrossRaw === 'number' ? amountGrossRaw : Number(amountGrossRaw);
      const amount = Number(payment.amount || amountGross || 0);

      if (payment.type === 'booking' && payment.bookingId) {
        await db.collection('bookings').doc(payment.bookingId).set(
          {
            paymentStatus: 'Paid',
            paymentMethod: 'PayFast',
            paymentAmount: amount,
            paidAt: admin.firestore.FieldValue.serverTimestamp(),
          },
          { merge: true },
        );
      }

      if (payment.type === 'wallet_topup' && payment.uid) {
        await db.collection('users').doc(payment.uid).set(
          {
            walletBalance: admin.firestore.FieldValue.increment(amount),
          },
          { merge: true },
        );
      }
    }

    return res.status(200).send('OK');
  } catch (err) {
    console.error('PayFast ITN error:', err);
    return res.status(500).send('ERROR');
  }
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
    const {
      cnicFrontBase64,
      cnicBackBase64,
      cnicFrontUrl,
      cnicBackUrl,
      expectedName,
      expectedFatherName,
      expectedDob,
    } = req.body || {};

    let resolvedFrontBase64 = cnicFrontBase64;
    if (!resolvedFrontBase64 && cnicFrontUrl) {
      resolvedFrontBase64 = await downloadRemoteImageAsBase64(cnicFrontUrl, 'CNIC front image');
    }

    if (!resolvedFrontBase64) {
      return res.status(400).json({
        error: 'cnicFrontBase64 or cnicFrontUrl is required',
        errorCode: 'MISSING_CNIC_FRONT',
      });
    }

    console.log("Processing CNIC via Local Engine...");
    const frontResult = await runKycEngine('cnic', {
      image: normalizeBase64(resolvedFrontBase64),
    });

    let backResult = null;
    let backError = null;
    let resolvedBackBase64 = cnicBackBase64;
    if (!resolvedBackBase64 && cnicBackUrl) {
      try {
        resolvedBackBase64 = await downloadRemoteImageAsBase64(cnicBackUrl, 'CNIC back image');
      } catch (e) {
        resolvedBackBase64 = null;
        backError = e?.message ?? String(e);
        console.warn('CNIC back download failed (continuing with front only):', backError);
      }
    }

    if (resolvedBackBase64) {
      try {
        backResult = await runKycEngine('cnic', {
          image: normalizeBase64(resolvedBackBase64),
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
    const {
      cnicImage,
      selfieImage,
      cnicImageUrl,
      selfieImageUrl,
    } = req.body || {};

    let resolvedCnic = cnicImage;
    if (!resolvedCnic && cnicImageUrl) {
      resolvedCnic = await downloadRemoteImageAsBase64(cnicImageUrl, 'CNIC image');
    }

    let resolvedSelfie = selfieImage;
    if (!resolvedSelfie && selfieImageUrl) {
      resolvedSelfie = await downloadRemoteImageAsBase64(selfieImageUrl, 'selfie image');
    }

    if (!resolvedCnic || !resolvedSelfie) {
      return res.status(400).json({
        error: 'cnicImage/selfieImage (base64) or cnicImageUrl/selfieImageUrl is required',
        errorCode: 'MISSING_FACE_INPUTS',
      });
    }

    console.log("Verifying Face via Local Engine...");
    const faceResult = await runKycEngine('face', {
      image: normalizeBase64(resolvedCnic),
      image2: normalizeBase64(resolvedSelfie),
    });

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
    const { shopImage, shopImageUrl } = req.body || {};

    let resolvedShop = shopImage;
    if (!resolvedShop && shopImageUrl) {
      resolvedShop = await downloadRemoteImageAsBase64(shopImageUrl, 'shop image');
    }

    if (!resolvedShop) {
      return res.status(400).json({
        error: 'shopImage (base64) or shopImageUrl is required',
        errorCode: 'MISSING_SHOP_IMAGE',
      });
    }

    console.log("Verifying Shop via Local Engine...");
    const result = await runKycEngine('shop', { image: normalizeBase64(resolvedShop) });

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
