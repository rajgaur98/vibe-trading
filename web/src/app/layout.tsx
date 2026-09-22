import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import TopBar from "@/components/TopBar";
import { Providers } from "@/components/providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "Vibe Trading Portal — Automated Crypto Agentic Bot",
  description: "A crypto swing-trading system powered by Google Gemini multi-agent reasoning and deterministic risk rules.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} h-full antialiased`}
    >
      <body className="h-full bg-background font-sans text-foreground antialiased">
        <Providers>
          <div className="flex h-full flex-row overflow-hidden">
            <Navigation />
            <main className="flex h-full flex-1 flex-col overflow-hidden">
              <TopBar />
              <div className="flex-1 overflow-y-auto bg-muted/40">
                {children}
              </div>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
