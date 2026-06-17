# Design Spec — Frontend Vercel Deployment (Approach A)

> **Initiative:** Re-platform and deploy the Next.js dashboard to Vercel for zero-monthly-cost hosting, saving local memory/CPU resources on the Oracle Micro VM.

## 1. Goal

Expose the Next.js dashboard publicly on Vercel while routing all data queries (FastAPI endpoints) back to the `vibe-api` service running on the Oracle Always-Free Micro VM. This saves ~150MB of RAM on the VM by avoiding a local `vibe-web` node service and takes advantage of Vercel's free global CDN, SSL certificates, and git-push-to-deploy capabilities.

## 2. Proposed Changes

### A. Next.js Routing config
No code changes are required in the Next.js frontend code because the existing `web/next.config.ts` already implements server-side rewrites using the `API_HOST` environment variable:
```typescript
  async rewrites() {
    const apiHost = process.env.API_HOST || "http://vibe-api:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiHost}/api/:path*`,
      },
    ];
  },
```

### B. Vercel Project Configuration
When deploying the `/web` subdirectory to Vercel:
* **Framework Preset**: Next.js (automatically detected).
* **Root Directory**: `web` (must be specified since the Next.js app lives in a subdirectory).
* **Environment Variables**:
  - `API_HOST` = `http://150.230.237.230:8008` (the public FastAPI endpoint of the Oracle VM).

### C. Oracle VM Security Configuration (Host-side)
To make the FastAPI endpoints accessible to Vercel's serverless functions, the VM host must allow incoming TCP traffic on port `8008`:
1. **Oracle Cloud Infrastructure Console (OCI)**: Add an Ingress Rule in the VCN's Security List:
   - Source CIDR: `0.0.0.0/0` (or Vercel IP range, but since Vercel IPs are dynamic, public `0.0.0.0/0` is preferred; FastAPI CORS is configured hermetically).
   - IP Protocol: `TCP`
   - Source Port Range: `All`
   - Destination Port Range: `8008`
2. **OS firewall (iptables / UFW)**: Open port `8008` on the VM host.

---

## 3. Data Flow

```
User Browser ──(HTTPS)──► Vercel Edge Serverless Function (/api/metrics)
                               │
                               ▼ (Server-side rewrite)
                     http://150.230.237.230:8008/api/metrics
                               │
                               ▼ (Public Internet TCP/8008)
                       Oracle Cloud Micro VM
                               │
                      Docker container (vibe_trading_api)
                               │
                               ├─► Read portfolio_state / trades / costs (Supabase Postgres)
                               └─► Read candles / active positions close prices (DuckDB cache)
```

---

## 4. Verification & Rollout Plan

### Automated Verification
* Run unit tests on the API server to confirm the FastAPI endpoints `/api/status`, `/api/metrics`, `/api/positions`, and `/api/costs` return structured JSON successfully.

### Manual Verification
1. Verify API server availability from the internet:
   - Send `curl http://150.230.237.230:8008/api/status` and ensure it responds with `{"status": "online", ...}`.
2. Deploy the Next.js dashboard to Vercel using the Vercel CLI or Git integration.
3. Access the Vercel deployment URL, verify the metrics dashboard loads, the charts populate, and the miniTicker WebSockets connect successfully in the browser console.
