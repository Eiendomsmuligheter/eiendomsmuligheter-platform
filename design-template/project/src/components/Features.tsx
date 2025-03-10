import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Zap, BarChart3, Calculator } from 'lucide-react';

const features = [
  {
    icon: <FileText className="w-8 h-8" />,
    title: 'Instant Property Reports',
    description: 'AI-generated analysis on ROI, zoning regulations, and profitability'
  },
  {
    icon: <Zap className="w-8 h-8" />,
    title: 'Automated Application Forms',
    description: 'Ready-to-use application templates for property permits'
  },
  {
    icon: <BarChart3 className="w-8 h-8" />,
    title: 'Energy & Sustainability Reports',
    description: 'AI-driven energy efficiency analysis'
  },
  {
    icon: <Calculator className="w-8 h-8" />,
    title: 'Smart Pricing Estimations',
    description: 'Data-driven valuation of properties'
  }
];

export const Features = () => {
  return (
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600 mb-4">
            Why Use This Platform?
          </h2>
          <p className="text-gray-400 text-xl">
            Powered by cutting-edge AI technology to maximize your property investments
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              className="p-6 rounded-2xl bg-gradient-to-br from-white/5 to-white/10 border border-white/10 backdrop-blur-lg hover:from-white/10 hover:to-white/15 transition-all"
            >
              <div className="text-blue-400 mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};