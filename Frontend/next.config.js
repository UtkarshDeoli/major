/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  webpack: (config) => {
    // Exclude canvas and other binary modules from being processed by webpack
    config.externals = [...(config.externals || []), { canvas: "canvas" }];
    
    return config;
  },
};

module.exports = nextConfig;
