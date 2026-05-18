export function WarningsBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <aside
      role="status"
      className="glass flex gap-4 p-5 md:p-6"
      style={{
        background:
          "linear-gradient(135deg, rgba(239, 224, 187, 0.6), rgba(255, 253, 248, 0.7))",
      }}
    >
      <span
        aria-hidden
        className="grid h-10 w-10 flex-none place-items-center rounded-full text-base"
        style={{
          background: "linear-gradient(135deg, var(--ochre-soft), var(--paper-3))",
          color: "var(--ochre-deep)",
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.6), 0 6px 18px -10px rgba(182,134,51,0.55)",
        }}
      >
        !
      </span>
      <div className="flex flex-col gap-1.5">
        <p className="text-sm font-medium text-[color:var(--ochre-deep)]">Partial result</p>
        <ul className="flex flex-col gap-1 text-sm leading-relaxed text-[color:var(--ink-soft)]">
          {warnings.map((message, index) => (
            <li key={index} className="flex gap-2">
              <span aria-hidden className="text-[color:var(--ochre)]">·</span>
              <span>{message}</span>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
