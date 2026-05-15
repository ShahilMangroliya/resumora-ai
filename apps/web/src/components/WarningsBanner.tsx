export function WarningsBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <aside
      role="status"
      className="flex flex-col gap-2 rounded-xl border border-amber-300 bg-amber-50 p-4 text-amber-900 dark:border-amber-700/60 dark:bg-amber-900/30 dark:text-amber-100"
    >
      <p className="text-sm font-medium">Partial result</p>
      <ul className="list-disc pl-5 text-sm">
        {warnings.map((message, index) => (
          <li key={index}>{message}</li>
        ))}
      </ul>
    </aside>
  );
}
