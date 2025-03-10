import React from 'react';
import { Hero } from './components/Hero';
import { Features } from './components/Features';
import { ChatAssistant } from './components/ChatAssistant';

function App() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Hero />
      <Features />
      <ChatAssistant />
    </div>
  );
}

export default App;