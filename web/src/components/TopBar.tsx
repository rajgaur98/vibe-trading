"use client";

import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      className="text-muted-foreground hover:text-foreground"
    >
      {/* CSS-driven swap off the <html> theme class — no client-only state,
          so the server and first client render agree (no hydration flash). */}
      <Moon className="block dark:hidden" />
      <Sun className="hidden dark:block" />
    </Button>
  );
}

export default function TopBar() {
  return (
    <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center justify-end gap-1 border-b border-border bg-background/80 px-4 backdrop-blur-md sm:px-6">
      <ThemeToggle />
    </header>
  );
}
