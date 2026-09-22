import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

/**
 * Colored pill for an agent action/side. long → gain, short → loss,
 * flat/close → neutral. Shared by the dashboard feed and the decisions log.
 */
export default function ActionBadge({
  action,
  className,
}: {
  action: string;
  className?: string;
}) {
  const a = action.toLowerCase();
  if (a === "long")
    return <Badge className={cn("border-transparent bg-gain/10 text-gain uppercase", className)}>{action}</Badge>;
  if (a === "short")
    return <Badge className={cn("border-transparent bg-loss/10 text-loss uppercase", className)}>{action}</Badge>;
  return <Badge variant="secondary" className={cn("uppercase", className)}>{action}</Badge>;
}
