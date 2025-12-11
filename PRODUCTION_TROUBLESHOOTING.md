# Production 500 Error Troubleshooting

## Immediate Actions

### 1. Check Render Logs NOW
```
Render Dashboard → Your Service → Logs Tab
```

Look for these specific error patterns:

#### Authentication Errors:
```
✗ Auth failed: Invalid X-Client-Key
✗ Auth failed: Account is suspended  
✗ Auth failed: API key revoked
```
**Fix:** User needs to log in again. API key is invalid or expired.

#### Database Connection Errors:
```
Failed to connect to Supabase
Connection timeout
Database error
```
**Fix:** Check Supabase credentials in Render environment variables.

#### Missing Environment Variables:
```
KeyError: 'SUPABASE_URL'
'NoneType' object has no attribute
```
**Fix:** Set missing environment variables in Render.

### 2. Quick Health Check

Visit these URLs in your browser:

**Health Check:**
```
https://rfpbackend-vho7.onrender.com/health
```
Expected: `{"status":"healthy","database":"connected"}`

**Root Endpoint:**
```
https://rfpbackend-vho7.onrender.com/
```
Expected: `{"message":"RFP Backend API",...}`

If either fails → backend is not running or has configuration issues.

### 3. Test Authentication Manually

**Step A: Login**
```bash
curl -X POST https://rfpbackend-vho7.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@bidwell.com","password":"admin123"}'
```

**Step B: Use API Key from response**
```bash
# Replace YOUR_API_KEY with the api_key from step A
curl -H "X-Client-Key: YOUR_API_KEY" \
     https://rfpbackend-vho7.onrender.com/rfps
```

If Step A fails → Database/Supabase issue
If Step B fails → API key not being sent correctly from frontend

## Common Root Causes

### Issue 1: Render Service Sleeping (Free Tier)
**Symptom:** First request after inactivity returns 500 or times out
**Cause:** Free Render services spin down after 15 min of inactivity
**Solution:** 
- Wait 60 seconds for service to wake up, then try again
- OR upgrade to paid tier for always-on service

### Issue 2: Wrong Supabase Credentials
**Symptom:** All endpoints return 500, logs show database errors
**Fix:**
1. Go to Supabase Dashboard → Project Settings → API
2. Copy these EXACT values:
   - Project URL → `SUPABASE_URL`
   - `service_role` key (NOT anon key) → `SUPABASE_KEY`
3. Update in Render → Environment
4. Click "Manual Deploy" → "Deploy Latest Commit"

### Issue 3: Frontend Sending Wrong/No API Key
**Symptom:** Logs show "Missing X-Client-Key" or "Invalid X-Client-Key"
**Check:** 
1. Open browser DevTools (F12)
2. Go to Application → Local Storage
3. Check if `clientKey` exists and has a value
4. If missing → User not logged in
5. If present → Check Network tab, verify `X-Client-Key` header is being sent

**Fix:** User needs to log in at `/login`

### Issue 4: CORS Configuration
**Symptom:** Browser console shows CORS errors before 500 error
**Fix:**
In Render Environment Variables:
```
FRONTEND_ORIGIN=https://your-app.vercel.app
```
(Replace with your actual Vercel URL, no trailing slash)

Then restart the service.

### Issue 5: Database Migration Not Applied
**Symptom:** 500 errors on specific endpoints, logs show "column does not exist"
**Fix:**
Run migrations on your Supabase database:
```bash
# Connect to your Supabase via psql
psql postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres

# Run migrations
\i RFP-backend/migrations/2025-12-10_admin_dashboard.sql
\i RFP-backend/migrations/2025-12-11_fix_cascading_deletes.sql
```

## Debugging Steps

### Step 1: Verify Environment Variables in Render

Required variables:
```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbG... (service_role key)
SUPABASE_JWT_SECRET=your-jwt-secret
FRONTEND_ORIGIN=https://your-vercel-app.vercel.app
OPENAI_API_KEY=sk-... (if using OpenAI)
GEMINI_API_KEY=... (if using Gemini)
```

### Step 2: Check Recent Deployments

In Render Dashboard:
1. Go to Deployments tab
2. Check if latest deployment succeeded
3. If failed → Click to see build logs
4. Look for Python errors during deployment

### Step 3: Enable Debug Logging

Add to Render Environment:
```
LOG_LEVEL=DEBUG
```
Restart service, then check logs for more details.

### Step 4: Test Specific Endpoint

Based on error URL, test directly:

**For `/rfps` error:**
```bash
curl -v -H "X-Client-Key: YOUR_KEY" \
     https://rfpbackend-vho7.onrender.com/rfps
```

**For `/tenders/{id}` error:**
```bash
curl -v -H "X-Client-Key: YOUR_KEY" \
     https://rfpbackend-vho7.onrender.com/tenders/YOUR_TENDER_ID
```

The `-v` flag shows full request/response details.

## Frontend Debugging

### Check API Base URL

In your Vercel project:
1. Settings → Environment Variables
2. Verify: `VITE_API_BASE=https://rfpbackend-vho7.onrender.com`
3. NO trailing slash!

### Check Browser Console

1. Open DevTools (F12)
2. Console tab → Look for red errors
3. Network tab → Click failed request → Preview/Response tabs
4. Look for actual error message from backend

### Check localStorage

1. Application tab → Local Storage
2. Verify `clientKey` exists
3. If missing → user not authenticated

## Emergency Fixes

### Quick Rollback
If everything was working before:
1. Render Dashboard → Deployments
2. Find last working deployment
3. Click "..." → "Redeploy"

### Force Restart
Sometimes helps with connection issues:
1. Render Dashboard → Your Service
2. "Manual Deploy" → "Clear build cache & deploy"

### Temporary Bypass
To verify it's an auth issue, temporarily comment out auth in one endpoint:
```python
# In RFP-backend/api/rfps.py
@router.get("/rfps")
def list_rfps(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    # Temporarily skip auth for testing
    # client_id = get_client_id_from_key(x_client_key)
    client_id = "test-client-id"  # REMOVE THIS AFTER TESTING
```

⚠️ **NEVER deploy this to production long-term!**

## Getting Support

When asking for help, provide:
1. Full error from browser console (screenshot or text)
2. Last 100 lines from Render logs
3. Response from `/health` endpoint
4. What changed recently (deployment, config, etc.)
5. Steps to reproduce the error

## Monitoring

Set up monitoring to catch issues early:

**Render:**
- Dashboard → Notifications
- Set up Discord/Slack webhook for deployment failures

**Uptime Monitoring:**
- Use [UptimeRobot](https://uptimerobot.com/) (free)
- Monitor: `https://rfpbackend-vho7.onrender.com/health`
- Alert if down for > 2 minutes

**Logs:**
- Check Render logs daily
- Look for patterns of errors
- Set up log retention/export for debugging

