# Frontend Vercel Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the Next.js frontend dashboard to Vercel and route all data queries to the public FastAPI server running on the Oracle VM (port 8008).

**Architecture:** The Next.js app is deployed to Vercel. We configure the `API_HOST` environment variable to point to the Oracle VM. Vercel automatically proxies `/api/*` endpoints server-side to the Oracle VM, avoiding mixed-content security blocks.

**Tech Stack:** Next.js, Vercel, iptables (Ubuntu), curl.

---

## File Structure

No code changes are required in the `/web` or `/src` application files since routing and CORS middleware are already dynamically configured. The task focuses on host network security, Vercel deployment variables, and endpoint verification.

---

### Task 1: Open Port 8008 on the Oracle VM Host

**Files:**
- Modify: Host iptables rules (manual step on VM `150.230.237.230`)

- [ ] **Step 1: SSH into the Oracle VM**

  Run: `ssh -i ~/Downloads/ssh-key-2026-06-16.key ubuntu@150.230.237.230`

- [ ] **Step 2: Add iptables rule to allow TCP port 8008**

  Run on the VM host:
  ```bash
  sudo iptables -I INPUT 6 -p tcp --dport 8008 -j ACCEPT
  ```

- [ ] **Step 3: Persist the iptables changes**

  Run on the VM host:
  ```bash
  sudo netfilter-persistent save
  ```

- [ ] **Step 4: Verify that port 8008 is open**

  Run on the VM host:
  ```bash
  sudo iptables -L INPUT -n --line-numbers | grep 8008
  ```
  Expected: Output showing the ACCEPT rule for dport 8008 at line 6 or similar.

---

### Task 2: Verify FastAPI Server Public Accessibility & CORS

**Files:**
- Test: CLI query from the local laptop

- [ ] **Step 1: Test status endpoint from local shell**

  Run from your local laptop:
  ```bash
  curl -i http://150.230.237.230:8008/api/status
  ```
  Expected: HTTP/1.1 200 OK and JSON response containing `{"status":"online", ...}`.

- [ ] **Step 2: Test CORS response headers**

  Run from your local laptop:
  ```bash
  curl -i -H "Origin: http://example.com" -H "Access-Control-Request-Method: GET" -X OPTIONS http://150.230.237.230:8008/api/status
  ```
  Expected: Response includes the headers:
  - `access-control-allow-origin: *`
  - `access-control-allow-methods: *`

---

### Task 3: Deploy Dashboard to Vercel

**Files:**
- Deploy: `web` directory

- [ ] **Step 1: Run Vercel CLI deployment**

  Run from your local laptop inside `/Users/raj/vibe-trading/web`:
  ```bash
  npx -y vercel@latest --name vibe-trading-web
  ```
  (Follow the interactive prompt: Link to new project, accept default settings, but specify `Root Directory` as `web` if prompted, or link from the Vercel Web Dashboard).

- [ ] **Step 2: Set Vercel environment variables**

  In the Vercel Project Dashboard (Settings -> Environment Variables), add:
  * **Key**: `API_HOST`
  * **Value**: `http://150.230.237.230:8008`

- [ ] **Step 3: Trigger a new Vercel deployment**

  Run from your local laptop inside `/Users/raj/vibe-trading/web`:
  ```bash
  npx -y vercel@latest --prod
  ```
  Expected: Build succeeds, and Vercel outputs the production deployment URL.

- [ ] **Step 4: Verify the live deployment URL**

  Open the Vercel production URL in your browser. Verify:
  - The metrics cards populate with real trade data.
  - The live positions table displays.
  - No connection/CORS errors appear in the browser console.
