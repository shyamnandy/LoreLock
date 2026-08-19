import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // pin the trace root to this directory. Without it Next walks up and finds
  // a stray package-lock.json in the home directory, then bundles from there.
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
