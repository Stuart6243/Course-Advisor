import React, { useState, useRef, useEffect } from 'react';
import { 
  GraduationCap, Settings, Plus, Bot, ArrowUp, Copy, 
  Download, FileText, Database, X, ChevronDown, Upload, ArrowDown, Info
} from 'lucide-react';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: React.ReactNode;
  time: string;
};

const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    role: 'assistant',
    content: "Hello! I'm your AI Course Advisor. How can I help you plan your learning journey today?",
    time: '10:02 AM'
  },
  {
    id: '2',
    role: 'user',
    content: "Can you create a schedule for the Introduction to Machine Learning course?",
    time: '10:04 AM'
  },
  {
    id: '3',
    role: 'assistant',
    content: (
      <>
        <p>Certainly. Here is a proposed weekly schedule for the <strong>Intro to ML</strong> course based on your current progress and goals.</p>
        <div className="rounded-xl overflow-hidden border border-border-dark bg-[#0d1117] mt-4 mb-4 group/code-block">
          <div className="flex items-center justify-between px-4 py-2 bg-[#161b22] border-b border-border-dark">
            <span className="text-xs text-slate-400 font-mono">json_schedule_v1.json</span>
            <button className="text-xs flex items-center gap-1 text-slate-400 hover:text-white transition-colors">
              <Copy className="w-3.5 h-3.5" />
              Copy
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-border-dark">
                  <th className="px-4 py-3 text-xs font-medium text-slate-400 uppercase tracking-wider w-24">Week</th>
                  <th className="px-4 py-3 text-xs font-medium text-slate-400 uppercase tracking-wider">Topic Focus</th>
                  <th className="px-4 py-3 text-xs font-medium text-slate-400 uppercase tracking-wider text-right">Est. Time</th>
                </tr>
              </thead>
              <tbody className="text-sm font-mono text-slate-300">
                <tr className="hover:bg-[#161b22]/50 transition-colors border-b border-border-dark/50">
                  <td className="px-4 py-3 text-primary">01</td>
                  <td className="px-4 py-3">Linear Algebra Basics</td>
                  <td className="px-4 py-3 text-right text-emerald-400">5h 30m</td>
                </tr>
                <tr className="hover:bg-[#161b22]/50 transition-colors border-b border-border-dark/50">
                  <td className="px-4 py-3 text-primary">02</td>
                  <td className="px-4 py-3">Python for Data Science</td>
                  <td className="px-4 py-3 text-right text-emerald-400">6h 00m</td>
                </tr>
                <tr className="hover:bg-[#161b22]/50 transition-colors border-b border-border-dark/50">
                  <td className="px-4 py-3 text-primary">03</td>
                  <td className="px-4 py-3">Supervised Learning Algorithms</td>
                  <td className="px-4 py-3 text-right text-emerald-400">8h 15m</td>
                </tr>
                <tr className="hover:bg-[#161b22]/50 transition-colors">
                  <td className="px-4 py-3 text-primary">04</td>
                  <td className="px-4 py-3">Model Evaluation & Validation</td>
                  <td className="px-4 py-3 text-right text-emerald-400">4h 45m</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <p>Would you like me to adjust the intensity of any specific week, or should we proceed with this plan?</p>
      </>
    ),
    time: '10:04 AM'
  }
];

function AutoResizeTextarea({ 
  value, 
  onChange, 
  onKeyDown, 
  placeholder, 
  className, 
  minHeight 
}: { 
  value: string, 
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void, 
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void, 
  placeholder: string, 
  className: string, 
  minHeight: string 
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={onChange}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      className={className}
      rows={1}
      style={{ minHeight }}
    />
  );
}

function Header({ onNewChat, onOpenSettings }: { onNewChat: () => void, onOpenSettings: () => void }) {
  return (
    <header className="flex-none flex items-center justify-between whitespace-nowrap border-b border-border-dark px-4 md:px-6 py-3 bg-background-dark z-20">
      <div className="flex items-center gap-3 text-white">
        <div className="w-8 h-8 flex items-center justify-center text-primary">
          <GraduationCap className="w-7 h-7" />
        </div>
        <h2 className="text-lg font-bold leading-tight tracking-[-0.015em]">Course Advisor AI</h2>
      </div>
      <div className="flex items-center gap-4">
        <button 
          onClick={onNewChat}
          className="hidden md:flex min-w-[100px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-9 px-4 bg-primary text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-blue-600 transition-colors"
        >
          <span className="truncate">New Chat</span>
          <Plus className="w-4 h-4 ml-2" />
        </button>
        <button onClick={onOpenSettings} className="w-9 h-9 rounded-full overflow-hidden ring-2 ring-border-dark hover:ring-primary transition-colors">
          <img src="https://picsum.photos/seed/avatar/100/100" alt="User" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
        </button>
      </div>
    </header>
  );
}

function LandingView({ 
  inputText, 
  setInputText, 
  onStart 
}: { 
  inputText: string, 
  setInputText: (val: string) => void, 
  onStart: () => void 
}) {
  return (
    <main className="flex-1 flex flex-col items-center justify-center w-full max-w-5xl mx-auto px-4 sm:px-6 relative">
      <div className="mb-8 flex justify-center">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-transparent flex items-center justify-center ring-1 ring-white/5 shadow-[0_0_20px_-5px_rgba(25,127,230,0.3)] backdrop-blur-sm">
          <GraduationCap className="w-8 h-8 text-primary" />
        </div>
      </div>
      <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-center mb-8 bg-gradient-to-b from-white to-slate-400 bg-clip-text text-transparent pb-1">
        Where should we begin?
      </h1>
      <div className="w-full max-w-[820px] relative">
        <div className="relative flex flex-col w-full bg-[#1e2024] border border-border-dark rounded-full focus-within:shadow-[0_0_20px_-5px_rgba(25,127,230,0.3)] focus-within:border-primary/50 transition-all duration-300 overflow-hidden">
          <AutoResizeTextarea 
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                onStart();
              }
            }}
            className="w-full bg-transparent text-base md:text-lg text-white placeholder-slate-500 px-6 py-4 pr-14 focus:outline-none resize-none max-h-[200px] overflow-y-auto leading-relaxed" 
            placeholder="Ask anything regarding your course path..." 
            minHeight="60px"
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-2">
            <button 
              onClick={onStart}
              disabled={!inputText.trim()}
              className="flex items-center justify-center w-9 h-9 rounded-full bg-[#2a2d33] text-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:bg-primary hover:text-white group/send"
            >
              <ArrowUp className="w-5 h-5 translate-y-0.5 group-hover/send:-translate-y-0.5 transition-transform" />
            </button>
          </div>
        </div>
      </div>
      <footer className="absolute bottom-4 py-4 px-6 text-center w-full">
        <p className="text-xs text-slate-600">
          Course Advisor AI can make mistakes. Consider checking important information.
        </p>
      </footer>
    </main>
  );
}

function ChatView({ 
  messages, 
  inputText, 
  setInputText, 
  onSend 
}: { 
  messages: Message[], 
  inputText: string, 
  setInputText: (val: string) => void, 
  onSend: () => void 
}) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <>
      <main className="flex-1 overflow-y-auto relative scroll-smooth pb-32" id="chat-container">
        <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col gap-6">
          {messages.map((msg, idx) => (
            <div 
              key={msg.id} 
              className={`flex gap-4 p-2 animate-fade-in ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              style={{ animationDelay: `${Math.min(idx * 0.1, 0.5)}s` }}
            >
              <div className="flex-none">
                {msg.role === 'assistant' ? (
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary ring-1 ring-primary/30">
                    <Bot className="w-5 h-5" />
                  </div>
                ) : (
                  <img src="https://picsum.photos/seed/avatar/100/100" alt="User" className="w-8 h-8 rounded-full object-cover" referrerPolicy="no-referrer" />
                )}
              </div>
              <div className={`flex-1 space-y-2 flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`flex items-baseline gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <span className="font-bold text-sm">{msg.role === 'assistant' ? 'Advisor AI' : 'You'}</span>
                  <span className="text-xs text-slate-500">{msg.time}</span>
                </div>
                <div className={`${msg.role === 'user' ? 'bg-surface-dark text-slate-100 rounded-2xl rounded-tr-sm px-5 py-3 max-w-[85%]' : 'text-slate-300 w-full'} text-[15px] leading-[1.55]`}>
                  {msg.content}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} className="h-12" />
        </div>
      </main>

      {/* Scroll to bottom button */}
      <div className="fixed bottom-28 right-8 z-30">
        <button 
          onClick={() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })}
          className="w-10 h-10 rounded-full bg-border-dark border border-slate-700 text-slate-300 shadow-lg hover:bg-slate-700 hover:text-white flex items-center justify-center transition-all group"
        >
          <ArrowDown className="w-5 h-5 group-hover:translate-y-0.5 transition-transform" />
        </button>
      </div>

      {/* Chat Input Area */}
      <div className="fixed bottom-0 left-0 w-full bg-gradient-to-t from-background-dark via-background-dark/95 to-transparent pt-10 pb-6 px-4 z-40 pointer-events-none">
        <div className="max-w-3xl mx-auto relative pointer-events-auto">
          <div className="relative bg-surface-dark border border-border-dark rounded-xl shadow-2xl focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-primary transition-all duration-200">
            <AutoResizeTextarea 
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  onSend();
                }
              }}
              className="w-full bg-transparent text-slate-200 placeholder-slate-500 text-[15px] p-4 pr-12 rounded-xl focus:ring-0 focus:outline-none resize-none overflow-hidden max-h-48" 
              placeholder="Ask Advisor AI..." 
              minHeight="56px"
            />
            <button 
              onClick={onSend}
              disabled={!inputText.trim()}
              className="absolute bottom-2 right-2 p-1.5 rounded-lg bg-primary text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowUp className="w-5 h-5" />
            </button>
          </div>
          <div className="text-center mt-2">
            <p className="text-[11px] text-slate-500">
              Advisor AI can make mistakes. Consider checking important information.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

function SettingsDrawer({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) {
  return (
    <div 
      className={`fixed inset-0 z-50 flex justify-end transition-all duration-300 ${isOpen ? 'opacity-100 visible' : 'opacity-0 invisible'}`}
    >
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Drawer */}
      <div 
        className={`relative w-full max-w-[480px] h-full bg-drawer-bg border-l border-border-dark shadow-2xl flex flex-col transition-transform duration-300 ease-out ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-5 border-b border-border-dark shrink-0">
          <h2 className="text-white text-lg font-bold leading-tight tracking-[-0.015em]">Settings</h2>
          <button 
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors rounded-full p-1 hover:bg-[#293038] flex items-center justify-center"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col gap-8">
          
          {/* Section 1: Language */}
          <section className="flex flex-col gap-3">
            <label className="text-slate-400 text-xs font-bold uppercase tracking-wider" htmlFor="language-select">Interface Language</label>
            <div className="relative">
              <select 
                id="language-select"
                className="w-full appearance-none rounded-xl bg-input-bg border border-[#3c4753] text-white px-4 py-3 pr-10 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              >
                <option value="en">English (US)</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-400">
                <ChevronDown className="w-5 h-5" />
              </div>
            </div>
          </section>

          <hr className="border-t border-border-dark" />

          {/* Section 2: Import Files */}
          <section className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-xs font-bold uppercase tracking-wider">Import Knowledge Base</span>
              <span className="text-slate-500 text-xs" title="Supported formats: PDF, TXT, MD">
                <Info className="w-4 h-4" />
              </span>
            </div>
            <div className="group relative flex flex-col items-center gap-4 rounded-xl border-2 border-dashed border-[#3c4753] hover:border-slate-500 hover:bg-input-bg transition-all px-6 py-10">
              <div className="flex flex-col items-center gap-2 text-center">
                <div className="p-3 bg-[#293038] rounded-full text-slate-300 group-hover:text-white group-hover:bg-primary/20 transition-colors">
                  <Upload className="w-6 h-6" />
                </div>
                <p className="text-white text-sm font-medium">Drag & drop files here</p>
                <p className="text-slate-400 text-xs">Limit 25MB per file • PDF, TXT, MD</p>
              </div>
              <button className="mt-2 flex items-center justify-center rounded-lg bg-[#293038] hover:bg-[#363f4a] text-white text-sm font-semibold px-4 py-2 transition-colors">
                Choose file
              </button>
              <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" />
            </div>
          </section>

          <hr className="border-t border-border-dark" />

          {/* Section 4: Export */}
          <section className="flex flex-col gap-4 pb-6">
            <span className="text-slate-400 text-xs font-bold uppercase tracking-wider">Export Chat History</span>
            <div className="flex gap-4">
              <label className="flex-1 flex items-center gap-3 p-3 rounded-xl border border-[#3c4753] cursor-pointer hover:bg-input-bg has-[:checked]:border-white transition-all">
                <input type="radio" name="export_format" value="markdown" defaultChecked className="peer h-4 w-4 border-slate-500 text-white bg-transparent focus:ring-0 checked:bg-white checked:border-white" />
                <span className="text-white text-sm font-medium">Markdown</span>
                <FileText className="w-5 h-5 text-slate-500 ml-auto" />
              </label>
              <label className="flex-1 flex items-center gap-3 p-3 rounded-xl border border-[#3c4753] cursor-pointer hover:bg-input-bg has-[:checked]:border-white transition-all">
                <input type="radio" name="export_format" value="json" className="peer h-4 w-4 border-slate-500 text-white bg-transparent focus:ring-0 checked:bg-white checked:border-white" />
                <span className="text-white text-sm font-medium">JSON</span>
                <Database className="w-5 h-5 text-slate-500 ml-auto" />
              </label>
            </div>
            <button className="w-full mt-2 flex items-center justify-center gap-2 rounded-xl bg-primary hover:bg-blue-600 text-white font-bold h-12 transition-colors shadow-lg shadow-blue-900/20">
              <Download className="w-5 h-5" />
              <span>Export Data</span>
            </button>
          </section>

        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border-dark text-center">
          <p className="text-xs text-slate-600">v2.4.0 • Build 8839a</p>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [inputText, setInputText] = useState('');
  
  const hasStarted = messages.length > 0;

  const handleSend = () => {
    if (!inputText.trim()) return;
    
    const newMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setMessages([...messages, newMessage]);
    setInputText('');
  };

  const handleNewChat = () => {
    setMessages([]);
    setInputText('');
  };

  return (
    <div className="min-h-screen flex flex-col overflow-hidden bg-background-dark text-slate-100 font-display selection:bg-primary/30 selection:text-white">
      <Header 
        onNewChat={handleNewChat} 
        onOpenSettings={() => setIsSettingsOpen(true)} 
      />

      {hasStarted ? (
        <ChatView 
          messages={messages}
          inputText={inputText} 
          setInputText={setInputText} 
          onSend={handleSend}
        />
      ) : (
        <LandingView 
          inputText={inputText} 
          setInputText={setInputText} 
          onStart={handleSend} 
        />
      )}

      <SettingsDrawer isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  );
}
