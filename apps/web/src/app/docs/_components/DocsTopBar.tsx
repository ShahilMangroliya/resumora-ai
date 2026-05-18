import Link from "next/link";

export function DocsTopBar() {
  return (
    <div className="flex items-center justify-between">
      <Link href="/" className="flex items-center gap-3">
        <span
          aria-hidden
          className="inline-grid h-9 w-9 place-items-center rounded-2xl border border-white/15 shadow-[inset_0_1px_0_rgba(255,255,255,0.22),0_8px_18px_-10px_rgba(14,93,82,0.5)]"
          style={{
            background:
              "linear-gradient(135deg, var(--teal) 0%, var(--teal-deep) 100%)",
          }}
        >
          <span className="italic-display text-lg leading-none text-[color:var(--paper-3)]">r</span>
        </span>
        <span className="font-display text-xl tracking-tight text-[color:var(--ink)]">
          Resumora
        </span>
      </Link>
      <nav className="hidden items-center gap-1 md:flex">
        <NavItem href="/">Analyze</NavItem>
        <NavItem href="/docs" active>Docs</NavItem>
        <NavItem href="https://github.com/anthropics/claude-code" external>Source</NavItem>
      </nav>
      <span className="pill"><span className="dot" /> Shahil</span>
    </div>
  );
}

function NavItem({
  children,
  href,
  active = false,
  external = false,
}: {
  children: React.ReactNode;
  href: string;
  active?: boolean;
  external?: boolean;
}) {
  const cls = `rounded-full px-4 py-1.5 text-sm transition-colors ${
    active
      ? "bg-white/60 text-[color:var(--ink)] shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_4px_14px_-8px_rgba(20,19,14,0.18)]"
      : "text-[color:var(--ink-soft)] hover:bg-white/40 hover:text-[color:var(--ink)]"
  }`;
  if (external) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={cls}>
        {children}
      </a>
    );
  }
  return (
    <Link href={href} className={cls}>
      {children}
    </Link>
  );
}
