import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Edge Vision — Enterprise Video Kinematics & AI Violence Detection",
  description:
    "Enterprise video kinematics and violence detection platform supporting asynchronous cloud batch analysis and high-throughput real-time edge execution.",
  keywords: [
    "violence detection", "edge AI", "video analysis", "ONNX Runtime",
    "YOLOv8-Pose", "LSTM", "biomechanical analysis", "kinematics",
    "batch processing", "real-time inference", "privacy-first",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased dark`}
    >
      <head>
        <meta name="theme-color" content="#09090b" />
        <meta name="color-scheme" content="dark" />
      </head>
      <body className="min-h-full flex flex-col bg-[#09090b] text-[#fafafa]">
        {children}
      </body>
    </html>
  );
}
