"use client";

import { ThemeProvider } from "next-themes";

/**
 * Client-side theme provider. next-themes toggles the `light`/`dark` class on
 * <html> (matching the tokens in globals.css) and persists the choice.
 * Defaults to light — the design target is the light neutral theme — but the
 * dark tokens stay fully wired, so the top-bar toggle switches cleanly.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="light"
      enableSystem
      disableTransitionOnChange
    >
      {children}
    </ThemeProvider>
  );
}
