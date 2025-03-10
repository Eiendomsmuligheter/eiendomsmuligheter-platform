/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  swcMinify: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*'
      }
    ]
  },
  images: {
    domains: ['localhost'],
  },
  // Fremtidige implementasjoner
  // experimental: {
  //   appDir: false,
  // },
}

module.exports = nextConfig 