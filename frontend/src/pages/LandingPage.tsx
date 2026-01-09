import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Stethoscope,
  Shield,
  Clock,
  FileText,
  ArrowRight,
  CheckCircle2,
  Sparkles,
  BookOpen,
  MessageSquare,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

const staggerChildren = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

export function LandingPage() {
  return (
    <div className="min-h-screen bg-surface-50">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-surface-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
                <Stethoscope className="w-5 h-5 text-white" />
              </div>
              <span className="font-display font-bold text-xl text-surface-900">
                Qualified Health
              </span>
            </div>
            
            <nav className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-sm font-medium text-surface-600 hover:text-primary-600 transition-colors">
                Features
              </a>
              <a href="#how-it-works" className="text-sm font-medium text-surface-600 hover:text-primary-600 transition-colors">
                How It Works
              </a>
              <a href="#guidelines" className="text-sm font-medium text-surface-600 hover:text-primary-600 transition-colors">
                Guidelines
              </a>
            </nav>
            
            <Link to="/chat" className="btn-primary">
              <MessageSquare className="w-4 h-4" />
              Start Chatting
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 mesh-bg" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="initial"
            animate="animate"
            variants={staggerChildren}
            className="text-center max-w-4xl mx-auto"
          >
            <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100 text-primary-700 text-sm font-medium mb-8">
              <Sparkles className="w-4 h-4" />
              AI-Powered Clinical Decision Support
            </motion.div>
            
            <motion.h1
              variants={fadeInUp}
              className="font-display font-extrabold text-5xl sm:text-6xl lg:text-7xl tracking-tight text-balance"
            >
              <span className="text-surface-900">Navigate Cancer Pathways</span>
              <br />
              <span className="gradient-text from-primary-600 via-trust-500 to-accent-500 bg-[length:200%_auto] animate-gradient">
                with Confidence
              </span>
            </motion.h1>
            
            <motion.p
              variants={fadeInUp}
              className="mt-8 text-xl text-surface-600 max-w-2xl mx-auto text-balance"
            >
              Evidence-based guidance for suspected cancer recognition and referral,
              grounded in NICE NG12 guideline. Covers 12+ cancer types, from lung to skin.
            </motion.p>
            
            <motion.div variants={fadeInUp} className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                to="/chat"
                className="btn-primary text-lg px-8 py-4 rounded-2xl group"
              >
                Try the Assistant
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <a
                href="#how-it-works"
                className="btn-secondary text-lg px-8 py-4 rounded-2xl"
              >
                Learn More
              </a>
            </motion.div>
            
            <motion.div variants={fadeInUp} className="mt-12 flex items-center justify-center gap-8 text-sm text-surface-500">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-accent-500" />
                NICE Compliant
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-accent-500" />
                Always Cited
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-accent-500" />
                Fail-Safe Design
              </div>
            </motion.div>
          </motion.div>
          
          {/* Hero illustration */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="mt-20 relative"
          >
            <div className="absolute inset-0 bg-gradient-to-t from-surface-50 via-transparent to-transparent z-10 pointer-events-none" />
            <div className="relative mx-auto max-w-5xl">
              <div className="card p-6 shadow-2xl shadow-primary-500/10">
                <div className="flex items-center gap-3 pb-4 border-b border-surface-100">
                  <div className="w-3 h-3 rounded-full bg-red-400" />
                  <div className="w-3 h-3 rounded-full bg-yellow-400" />
                  <div className="w-3 h-3 rounded-full bg-green-400" />
                  <span className="ml-4 text-sm text-surface-400 font-mono">Qualified Health Assistant</span>
                </div>
                <div className="pt-6 space-y-6">
                  <div className="flex gap-4">
                    <div className="w-10 h-10 rounded-full bg-surface-100 flex items-center justify-center flex-shrink-0">
                      <span className="text-lg">👤</span>
                    </div>
                    <div className="bg-surface-100 rounded-2xl rounded-tl-md px-4 py-3 max-w-md">
                      <p className="text-surface-700">
                        A 58-year-old patient presents with unintentional weight loss and persistent heartburn. What is the required referral pathway?
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-4">
                    <div className="w-10 h-10 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
                      <Stethoscope className="w-5 h-5 text-primary-600" />
                    </div>
                    <div className="bg-primary-50 rounded-2xl rounded-tl-md px-4 py-3 max-w-lg border border-primary-100">
                      <p className="text-surface-700">
                        Per 
                        <span className="inline-flex items-center mx-1 px-2 py-0.5 rounded-md bg-primary-100 text-primary-700 text-sm font-medium">
                          NG12 1.2.7
                        </span>
                        age ≥55 + weight loss + reflux → suspected cancer pathway referral.
                      </p>
                      <p className="mt-2 text-surface-700">
                        <strong>Action:</strong> 2-week wait referral for upper GI endoscopy.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center max-w-3xl mx-auto mb-16"
          >
            <h2 className="font-display font-bold text-4xl text-surface-900 mb-4">
              Built for Clinical Confidence
            </h2>
            <p className="text-lg text-surface-600">
              Every feature designed with healthcare professionals in mind, ensuring accurate, 
              auditable, and actionable guidance.
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Shield,
                title: '12+ Cancer Types',
                description: 'Lung, upper/lower GI, breast, urological, skin, head & neck, gynaecological, haematological, brain, sarcomas, and childhood cancers.',
                color: 'primary',
              },
              {
                icon: FileText,
                title: 'NG12 Citations',
                description: 'Every answer references specific NG12 recommendations (e.g., 1.3.1 for FIT testing). Full traceability for governance.',
                color: 'accent',
              },
              {
                icon: Clock,
                title: '2WW Pathways',
                description: 'Instant guidance on suspected cancer pathway referrals, urgent investigations, and direct access tests.',
                color: 'trust',
              },
              {
                icon: Zap,
                title: 'FIT & Investigations',
                description: 'Quantitative FIT testing criteria, chest X-ray indications, PSA guidance, and imaging pathways.',
                color: 'primary',
              },
              {
                icon: BookOpen,
                title: 'Safety Netting',
                description: 'Built-in guidance on follow-up for negative results, persistent symptoms, and escalation criteria.',
                color: 'accent',
              },
              {
                icon: MessageSquare,
                title: 'Symptom-Based Search',
                description: 'Ask by symptom (haematuria, dysphagia, lump) or by suspected cancer site. Natural clinical language.',
                color: 'trust',
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="card p-6 group"
              >
                <div
                  className={cn(
                    'w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110',
                    feature.color === 'primary' && 'bg-primary-100',
                    feature.color === 'accent' && 'bg-accent-100',
                    feature.color === 'trust' && 'bg-trust-100'
                  )}
                >
                  <feature.icon
                    className={cn(
                      'w-6 h-6',
                      feature.color === 'primary' && 'text-primary-600',
                      feature.color === 'accent' && 'text-accent-600',
                      feature.color === 'trust' && 'text-trust-600'
                    )}
                  />
                </div>
                <h3 className="font-display font-semibold text-lg text-surface-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-surface-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-24 bg-surface-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center max-w-3xl mx-auto mb-16"
          >
            <h2 className="font-display font-bold text-4xl text-surface-900 mb-4">
              Three Response Modes
            </h2>
            <p className="text-lg text-surface-600">
              The assistant adapts its response based on your query, ensuring appropriate and safe guidance.
            </p>
          </motion.div>
          
          <div className="grid lg:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Grounded Answers',
                subtitle: 'For in-scope queries',
                description: 'When you ask about referral criteria, timing requirements, or documentation needs, you get a complete answer with QS124 citations.',
                example: '"What are the criteria for urgent upper GI endoscopy?"',
                color: 'primary',
              },
              {
                step: '02',
                title: 'Smart Probing',
                subtitle: 'For under-specified queries',
                description: 'If critical information is missing (like patient age for certain pathways), the assistant asks clarifying questions before answering.',
                example: '"Should I order a FIT test?" → "What is the patient\'s age?"',
                color: 'accent',
              },
              {
                step: '03',
                title: 'Safe Refusals',
                subtitle: 'For out-of-scope queries',
                description: 'Treatment recommendations, diagnostic interpretation, and patient-specific advice are politely declined with explanations.',
                example: '"What treatment should I start?" → Redirected to in-scope alternatives',
                color: 'trust',
              },
            ].map((mode, index) => (
              <motion.div
                key={mode.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.15 }}
                className="relative"
              >
                <div className="card p-8 h-full">
                  <div
                    className={cn(
                      'text-6xl font-display font-extrabold mb-4 opacity-10',
                      mode.color === 'primary' && 'text-primary-500',
                      mode.color === 'accent' && 'text-accent-500',
                      mode.color === 'trust' && 'text-trust-500'
                    )}
                  >
                    {mode.step}
                  </div>
                  <h3 className="font-display font-bold text-xl text-surface-900 mb-1">
                    {mode.title}
                  </h3>
                  <p
                    className={cn(
                      'text-sm font-medium mb-4',
                      mode.color === 'primary' && 'text-primary-600',
                      mode.color === 'accent' && 'text-accent-600',
                      mode.color === 'trust' && 'text-trust-600'
                    )}
                  >
                    {mode.subtitle}
                  </p>
                  <p className="text-surface-600 mb-4">
                    {mode.description}
                  </p>
                  <div className="bg-surface-100 rounded-lg px-4 py-3">
                    <p className="text-sm text-surface-500 italic">
                      {mode.example}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Guidelines Section */}
      <section id="guidelines" className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="font-display font-bold text-4xl text-surface-900 mb-6">
                Powered by NICE NG12
              </h2>
              <p className="text-lg text-surface-600 mb-8">
                Comprehensive coverage of suspected cancer recognition and referral pathways 
                from the NICE NG12 guideline, updated May 2025.
              </p>
              
              <div className="space-y-4">
                {[
                  { id: '1.1-1.2', title: 'Lung, Pleural & Upper GI Cancers' },
                  { id: '1.3', title: 'Colorectal & Anal (FIT Testing)' },
                  { id: '1.4-1.6', title: 'Breast, Gynaecological & Urological' },
                  { id: '1.7-1.12', title: 'Skin, Head/Neck, Brain, Haematological' },
                  { id: '1.14-1.16', title: 'Safety Netting & Patient Support' },
                ].map((statement) => (
                  <div
                    key={statement.id}
                    className="flex items-center gap-4 p-4 rounded-xl bg-surface-50 border border-surface-200"
                  >
                    <div className="px-3 py-1 rounded-lg bg-primary-100 text-primary-700 font-mono text-sm font-medium">
                      {statement.id}
                    </div>
                    <span className="text-surface-700">{statement.title}</span>
                  </div>
                ))}
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary-200 via-trust-200 to-accent-200 rounded-3xl blur-3xl opacity-30" />
              <div className="relative card p-8">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center">
                    <BookOpen className="w-6 h-6 text-primary-600" />
                  </div>
                  <div>
                    <h3 className="font-display font-semibold text-lg text-surface-900">
                      NICE NG12
                    </h3>
                    <p className="text-sm text-surface-500">Suspected Cancer: Recognition & Referral</p>
                  </div>
                </div>
                
                <div className="prose prose-surface max-w-none">
                  <p className="text-surface-600">
                    Covers identifying children, young people and adults with symptoms 
                    that could be caused by cancer, with investigations and referral criteria.
                  </p>
                  <ul className="text-surface-600 space-y-2 mt-4">
                    <li>Published: June 2015</li>
                    <li>Last updated: May 2025</li>
                    <li>Covers: 12+ cancer types</li>
                  </ul>
                </div>
                
                <div className="mt-6 pt-6 border-t border-surface-200">
                  <p className="text-sm text-surface-500">
                    <strong className="text-surface-700">Note:</strong> This assistant is a 
                    decision-support tool and does not replace clinical judgment. Always 
                    consult the full guidelines for complex cases.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-primary-600 via-primary-700 to-trust-700">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="font-display font-bold text-4xl text-white mb-6">
              Ready to Navigate Pathways with Confidence?
            </h2>
            <p className="text-xl text-primary-100 mb-10 max-w-2xl mx-auto">
              Get instant, evidence-based guidance on suspected cancer recognition and referral. 
              Always cited, always safe.
            </p>
            <Link
              to="/chat"
              className="inline-flex items-center gap-3 bg-white text-primary-700 px-8 py-4 rounded-2xl font-semibold text-lg hover:bg-primary-50 transition-colors shadow-xl shadow-primary-900/20 group"
            >
              <MessageSquare className="w-5 h-5" />
              Start Your First Conversation
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-surface-900 text-surface-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-800 flex items-center justify-center">
                <Stethoscope className="w-5 h-5 text-primary-400" />
              </div>
              <span className="font-display font-bold text-lg text-white">
                Qualified Health
              </span>
            </div>
            
            <p className="text-sm text-center md:text-left">
              Clinical decision support based on NICE NG12. Not a substitute for professional medical judgment.
            </p>
            
            <p className="text-sm">
              © {new Date().getFullYear()} Qualified Health
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
