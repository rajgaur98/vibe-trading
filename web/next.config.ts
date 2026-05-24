import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const apiHost = process.env.API_HOST || "http://vibe-api:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiHost}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
