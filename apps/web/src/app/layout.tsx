import type { Metadata } from "next";
import { Fraunces, Instrument_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
  axes: ["opsz", "SOFT"],
  display: "swap",
});

const instrument = Instrument_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Resumora AI — Resume × Job-Description Fit Engine",
  description:
    "A calm, quiet way to ask how well your résumé answers a role — with a fit score, reasoning, and three rewrites you can lift straight into your CV.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${instrument.variable} ${jetbrains.variable} h-full antialiased`}
    >
      <body className="relative min-h-full overflow-x-hidden">
        <div aria-hidden className="pointer-events-none fixed inset-0 -z-30 bg-aurora" />
        <div aria-hidden className="pointer-events-none fixed inset-0 -z-20 bg-grain opacity-[0.5]" />
        {children}
      </body>
    </html>
  );
}
