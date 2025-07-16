import "./globals.css";

export const metadata = {
  title: "AI Coach Dashboard - Esports Coaching Platform",
  description:
    "Advanced AI-powered esports coaching and performance analysis system",
  generator: "v0.dev",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
      </head>
      <body className="bg-black text-white font-mono antialiased">
        {children}
      </body>
    </html>
  );
}
