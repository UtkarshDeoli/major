/** @type {import('next').NextConfig} */
const nextConfig = {
  // output: 'export',
  images: { unoptimized: true },
  turbopack: {},
  webpack: (config) => {
    config.externals = [...(config.externals || []), { canvas: "canvas" }];
    return config;
  },
};

module.exports = nextConfig;
