import React from 'react';
import { Hero } from '../components/Hero';
import { Features } from '../components/Features';
import { Pricing } from '../components/Pricing';
import { ChatAssistant } from '../components/ChatAssistant';

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Hero />
      <Features />
      <Pricing />
      <ChatAssistant />
    </div>
  );
} 