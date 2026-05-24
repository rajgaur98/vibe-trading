import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
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
      className={`${outfit.variable} dark h-full antialiased`}
    >
      <body className="h-full bg-slate-950 font-sans text-slate-200 antialiased flex flex-row overflow-hidden">
        {/* Navigation Sidebar */}
        <Navigation />

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col h-full overflow-y-auto bg-gradient-to-tr from-slate-950 via-slate-950 to-slate-900/40">
          {children}
        </main>
      </body>
    </html>
  );
}
