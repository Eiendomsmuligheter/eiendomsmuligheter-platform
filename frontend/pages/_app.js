import '../src/styles/globals.css';

export default function MyApp({ Component, pageProps }) {
  return (
    <main className="font-sans">
      <Component {...pageProps} />
    </main>
  );
} 