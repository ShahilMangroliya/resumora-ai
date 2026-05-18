export function WarningsBanner({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <aside
      role="status"
      className="glass flex gap-4 p-5 md:p-6"
      style={{
        background:
          "linear-gradient(135deg, rgba(243, 227, 168, 0.55), rgba(255, 255, 255, 0.7))",
      }}
    >
      <span
        aria-hidden
        className="grid h-10 w-10 flex-none place-items-center rounded-full text-base"
        style={{
          background: "linear-gradient(135deg, #f3e3a8, #fff)",
          color: "#8a6a1f",
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.7), 0 6px 18px -10px rgba(230,169,85,0.5)",
        }}
      >
        !
      </span>
      <div className="flex flex-col gap-1.5">
        <p className="text-sm font-medium text-[color:#8a6a1f]">Partial result</p>
        <ul className="flex flex-col gap-1 text-sm leading-relaxed text-[color:var(--ink-soft)]">
          {warnings.map((message, index) => (
            <li key={index} className="flex gap-2">
              <span aria-hidden className="text-[color:#c79a3f]">·</span>
              <span>{message}</span>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
