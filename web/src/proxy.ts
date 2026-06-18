import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Next.js 16 Proxy (formerly "middleware"). Injects the dashboard API key as an
 * upstream header on every `/api/*` request so the FastAPI backend on the Oracle VM
 * (reached via the `next.config.ts` rewrite to `API_HOST`) can authenticate the call.
 *
 * `NextResponse.next({ request: { headers } })` adds the header to what the *backend*
 * receives WITHOUT exposing it to the browser, so the secret never reaches the client.
 *
 * No-op when `DASHBOARD_API_KEY` is unset — the dashboard keeps working before the key
 * is rolled out, and only starts sending it once the env var is configured in Vercel.
 */
export function proxy(request: NextRequest) {
  const apiKey = process.env.DASHBOARD_API_KEY;
  if (!apiKey) {
    return NextResponse.next();
  }
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-api-key", apiKey);
  return NextResponse.next({ request: { headers: requestHeaders } });
}

export const config = {
  matcher: "/api/:path*",
};
