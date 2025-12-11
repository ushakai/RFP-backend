# Deployment Debugging Guide

## Current Issue: 500 Error on `/rfps` Endpoint

### Quick Diagnosis Steps

#### 1. Check Render Logs
In your Render dashboard, go to your service logs and look for:
- Authentication errors: `"✗ Auth failed:"`
- Database errors: `"✗ Database error listing RFPs:"`
- Connection errors: `"Failed to connect to Supabase"`

#### 2. Verify Environment Variables
In Render dashboard → Environment, ensure these are set:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_JWT_SECRET=your-jwt-secret
FRONTEND_ORIGIN=https://your-vercel-app.vercel.app
```

#### 3. Test Health Endpoint
Visit: `https://rfpbackend-vho7.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T...",
  "database": "connected"
}
```

If you get `"database": "disconnected"`, your Supabase credentials are wrong.

#### 4. Test with cURL
```bash
# Replace with a valid API key from your database
curl -H "X-Client-Key: YOUR_API_KEY_HERE" \
     https://rfpbackend-vho7.onrender.com/rfps
```

### Common Production Issues

#### Issue 1: Missing API Key
**Symptom:** `401 Unauthorized` or `"Missing X-Client-Key"`
**Solution:** 
- Check if user is logged in: Open browser DevTools → Application → LocalStorage → check `clientKey`
- If missing, user needs to log in again

#### Issue 2: Invalid Supabase Credentials
**Symptom:** `500 Internal Server Error` with database connection errors in logs
**Solution:**
1. Go to Supabase dashboard → Settings → API
2. Copy the correct values:
   - `SUPABASE_URL`: Project URL
   - `SUPABASE_KEY`: `service_role` key (not `anon` key)
3. Update in Render Environment variables
4. Restart the service

#### Issue 3: CORS Issues
**Symptom:** `CORS policy` errors in browser console
**Solution:**
- Set `FRONTEND_ORIGIN=https://your-vercel-app.vercel.app` in Render
- Restart the service

#### Issue 4: Timeout/Slow Response
**Symptom:** Request takes >30 seconds and times out
**Solution:**
- Check if Render service is on free tier (spins down after inactivity)
- First request after spin-down can take 30-60 seconds
- Consider upgrading to paid tier for always-on service

### Frontend Environment Variables (Vercel)

In Vercel dashboard → Settings → Environment Variables:
```
VITE_API_BASE=https://rfpbackend-vho7.onrender.com
```

After updating, redeploy the frontend.

### Manual Testing

1. **Test Authentication:**
```bash
curl -X POST https://rfpbackend-vho7.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@bidwell.com","password":"admin123"}'
```

Expected: `{"api_key":"...","email":"admin@bidwell.com","org_name":"...","role":"admin"}`

2. **Test RFPs with API Key:**
```bash
# Use the api_key from step 1
curl -H "X-Client-Key: YOUR_API_KEY" \
     https://rfpbackend-vho7.onrender.com/rfps
```

Expected: `{"rfps":[...]}`

### Logs Location

**Render:**
- Dashboard → Your Service → Logs tab
- Look for lines starting with `✓` (success) or `✗` (error)

**Browser:**
- DevTools (F12) → Console
- DevTools → Network → Find failed request → Preview/Response tabs

### Emergency Rollback

If you need to quickly rollback:
1. In Render dashboard → Deployments
2. Find the last working deployment
3. Click "..." → "Redeploy"

### Getting Help

Include these in your error report:
1. Full error message from browser console
2. Render service logs (last 50-100 lines)
3. Response from `/health` endpoint
4. Steps to reproduce

