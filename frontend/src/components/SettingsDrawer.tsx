import {type ChangeEvent, useEffect, useRef, useState} from 'react';
import {
  ChevronDown,
  Database,
  Download,
  FileText,
  Info,
  Upload,
  X,
} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {exportChat, importFile, importManual} from '../services/api';
import {ChatSettings, Language, ManualCourseData, Message} from '../types';
import ManualImportForm from './ManualImportForm';

type Props = {
  isOpen: boolean;
  onClose: () => void;
  language: Language;
  onLanguageChange: (lang: Language) => void;
  messages: Message[];
  maxHistoryTurns: number;
  maxResults: number;
  onSettingsChange: (settings: ChatSettings) => void;
};

type StatusTone = 'success' | 'error' | 'info';

type Notice = {
  tone: StatusTone;
  text: string;
};

export default function SettingsDrawer({
  isOpen,
  onClose,
  language,
  onLanguageChange,
  messages,
  maxHistoryTurns,
  maxResults,
  onSettingsChange,
}: Props) {
  const {t, i18n} = useTranslation();
  const [notice, setNotice] = useState<Notice | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [manualOpen, setManualOpen] = useState(false);
  const [manualPartial, setManualPartial] = useState<Partial<ManualCourseData> | undefined>();
  const [manualMissing, setManualMissing] = useState<string[] | undefined>();
  const [manualPreview, setManualPreview] = useState<string | undefined>();
  const [exportFormat, setExportFormat] = useState<'markdown' | 'json'>('markdown');
  const [isExporting, setIsExporting] = useState(false);
  const [exported, setExported] = useState(false);
  const [localMaxHistoryTurns, setLocalMaxHistoryTurns] = useState(maxHistoryTurns);
  const [localMaxResults, setLocalMaxResults] = useState(maxResults);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    setLocalMaxHistoryTurns(maxHistoryTurns);
    setLocalMaxResults(maxResults);
  }, [maxHistoryTurns, maxResults]);

  const clearNoticeLater = () => {
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = window.setTimeout(() => {
      setNotice(null);
    }, 3000);
  };

  const handleLanguage = (lang: Language) => {
    onLanguageChange(lang);
    i18n.changeLanguage(lang);
  };

  const clampInt = (value: number, min: number, max: number) => {
    if (!Number.isFinite(value)) {
      return min;
    }
    return Math.min(max, Math.max(min, Math.round(value)));
  };

  const applySettings = (next: ChatSettings, tone: StatusTone, text: string) => {
    onSettingsChange(next);
    setLocalMaxHistoryTurns(next.maxHistoryTurns);
    setLocalMaxResults(next.maxResults);
    setNotice({tone, text});
    clearNoticeLater();
  };

  const handleSaveSettings = () => {
    applySettings(
      {
        maxHistoryTurns: clampInt(localMaxHistoryTurns, 1, 50),
        maxResults: clampInt(localMaxResults, 1, 20),
      },
      'success',
      t('settings.settingsSaved'),
    );
  };

  const handleResetSettings = () => {
    applySettings(
      {
        maxHistoryTurns: 10,
        maxResults: 5,
      },
      'info',
      t('settings.settingsReset'),
    );
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) {
      return;
    }

    setIsImporting(true);
    setNotice({tone: 'info', text: t('settings.importProcessing')});

    try {
      const result = await importFile(file);
      if (result.success) {
        const courseLabel = result.course
          ? `${result.course.course_code} - ${result.course.title}`
          : '';
        setNotice({
          tone: 'success',
          text: `${t('settings.importSuccess')} ${courseLabel}`.trim(),
        });
        clearNoticeLater();
        return;
      }

      if (result.needs_manual_input) {
        setManualPartial(result.partial_data);
        setManualMissing(result.missing_fields);
        setManualPreview(result.extracted_text_preview);
        setManualOpen(true);
        setNotice({tone: 'error', text: result.message || t('settings.importError')});
        return;
      }

      setNotice({
        tone: 'error',
        text: `${t('settings.importError')}: ${result.message}`,
      });
      clearNoticeLater();
    } catch (err) {
      const msg = err instanceof Error ? err.message : t('settings.importError');
      setNotice({tone: 'error', text: `${t('settings.importError')}: ${msg}`});
      clearNoticeLater();
    } finally {
      setIsImporting(false);
    }
  };

  const handleManualSubmit = async (data: ManualCourseData) => {
    const result = await importManual(data);
    if (!result.success) {
      throw new Error(result.message || t('settings.importError'));
    }

    setManualOpen(false);
    setNotice({
      tone: 'success',
      text: `${t('settings.importSuccess')} ${result.course?.course_code ?? ''} ${result.course?.title ?? ''}`.trim(),
    });
    clearNoticeLater();
  };

  const handleExport = async () => {
    if (!messages.length || isExporting) {
      return;
    }

    setIsExporting(true);
    try {
      await exportChat(
        messages.map((msg) => ({role: msg.role, content: msg.content})),
        exportFormat,
      );
      setExported(true);
      window.setTimeout(() => setExported(false), 1500);
    } finally {
      setIsExporting(false);
    }
  };

  const noticeClass =
    notice?.tone === 'success'
      ? 'text-emerald-400'
      : notice?.tone === 'error'
        ? 'text-red-400'
        : 'text-slate-300';

  return (
    <div
      className={`fixed inset-0 z-50 flex justify-end transition-all duration-300 ${isOpen ? 'opacity-100 visible' : 'opacity-0 invisible'}`}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

      <div
        className={`relative w-full max-w-[480px] h-full bg-drawer-bg border-l border-border-dark shadow-2xl flex flex-col transition-transform duration-300 ease-out ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <header className="flex items-center justify-between px-6 py-5 border-b border-border-dark shrink-0">
          <h2 className="text-white text-lg font-bold leading-tight tracking-[-0.015em]">
            {t('settings.title')}
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors rounded-full p-1 hover:bg-[#293038] flex items-center justify-center"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 flex flex-col gap-8">
          <section className="flex flex-col gap-3">
            <label className="text-slate-400 text-xs font-bold uppercase tracking-wider" htmlFor="language-select">
              {t('settings.language')}
            </label>
            <div className="relative">
              <select
                id="language-select"
                value={language}
                onChange={(e) => handleLanguage(e.target.value as Language)}
                className="w-full appearance-none rounded-xl bg-input-bg border border-[#3c4753] text-white px-4 py-3 pr-10 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              >
                <option value="en">{t('languages.en')}</option>
                <option value="zh">{t('languages.zh')}</option>
                <option value="es">{t('languages.es')}</option>
                <option value="fr">{t('languages.fr')}</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-400">
                <ChevronDown className="w-5 h-5" />
              </div>
            </div>
          </section>

          <hr className="border-t border-border-dark" />

          <section className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-xs font-bold uppercase tracking-wider">
                {t('settings.querySettings')}
              </span>
              <button
                type="button"
                onClick={handleResetSettings}
                className="text-xs text-slate-500 hover:text-white transition-colors"
              >
                {t('settings.resetDefaults')}
              </button>
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-white text-sm font-medium" htmlFor="max-history-turns">
                {t('settings.maxHistoryTurns')}
              </label>
              <input
                id="max-history-turns"
                type="number"
                min={1}
                max={50}
                step={1}
                value={localMaxHistoryTurns}
                onChange={(e) => setLocalMaxHistoryTurns(Number(e.target.value || 1))}
                className="w-full rounded-xl bg-input-bg border border-[#3c4753] text-white px-4 py-3 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              />
              <p className="text-xs text-slate-500">{t('settings.maxHistoryTurnsHint')}</p>
            </div>

            <div className="flex flex-col gap-2">
              <label className="text-white text-sm font-medium" htmlFor="max-results">
                {t('settings.maxResults')}
              </label>
              <input
                id="max-results"
                type="number"
                min={1}
                max={20}
                step={1}
                value={localMaxResults}
                onChange={(e) => setLocalMaxResults(Number(e.target.value || 1))}
                className="w-full rounded-xl bg-input-bg border border-[#3c4753] text-white px-4 py-3 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
              />
              <p className="text-xs text-slate-500">{t('settings.maxResultsHint')}</p>
            </div>

            <button
              type="button"
              onClick={handleSaveSettings}
              className="w-full mt-1 rounded-xl bg-primary hover:bg-blue-600 text-white font-semibold h-11 transition-colors"
            >
              {t('settings.saveSettings')}
            </button>
          </section>

          <hr className="border-t border-border-dark" />

          <section className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-xs font-bold uppercase tracking-wider">
                {t('settings.import')}
              </span>
              <span className="text-slate-500 text-xs" title="Supported formats: PDF, HTML">
                <Info className="w-4 h-4" />
              </span>
            </div>
            <div className="group relative flex flex-col items-center gap-4 rounded-xl border-2 border-dashed border-[#3c4753] hover:border-slate-500 hover:bg-input-bg transition-all px-6 py-10">
              <div className="flex flex-col items-center gap-2 text-center">
                <div className="p-3 bg-[#293038] rounded-full text-slate-300 group-hover:text-white group-hover:bg-primary/20 transition-colors">
                  <Upload className="w-6 h-6" />
                </div>
                <p className="text-white text-sm font-medium">{t('settings.importDrag')}</p>
                <p className="text-slate-400 text-xs">{t('settings.importLimit')}</p>
              </div>
              <button
                type="button"
                className="mt-2 flex items-center justify-center rounded-lg bg-[#293038] hover:bg-[#363f4a] text-white text-sm font-semibold px-4 py-2 transition-colors"
              >
                {isImporting ? t('settings.importProcessing') : t('settings.importChoose')}
              </button>
              <input
                type="file"
                accept=".pdf,.html,.htm"
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={handleFileChange}
                disabled={isImporting}
              />
            </div>
            {notice ? <p className={`text-sm ${noticeClass}`}>{notice.text}</p> : null}
          </section>

          <hr className="border-t border-border-dark" />

          <section className="flex flex-col gap-4 pb-6">
            <span className="text-slate-400 text-xs font-bold uppercase tracking-wider">
              {t('settings.export')}
            </span>
            <div className="flex gap-4">
              <label className="flex-1 flex items-center gap-3 p-3 rounded-xl border border-[#3c4753] cursor-pointer hover:bg-input-bg has-[:checked]:border-white transition-all">
                <input
                  type="radio"
                  name="export_format"
                  value="markdown"
                  checked={exportFormat === 'markdown'}
                  onChange={() => setExportFormat('markdown')}
                  className="peer h-4 w-4 border-slate-500 text-white bg-transparent focus:ring-0 checked:bg-white checked:border-white"
                />
                <span className="text-white text-sm font-medium">Markdown</span>
                <FileText className="w-5 h-5 text-slate-500 ml-auto" />
              </label>
              <label className="flex-1 flex items-center gap-3 p-3 rounded-xl border border-[#3c4753] cursor-pointer hover:bg-input-bg has-[:checked]:border-white transition-all">
                <input
                  type="radio"
                  name="export_format"
                  value="json"
                  checked={exportFormat === 'json'}
                  onChange={() => setExportFormat('json')}
                  className="peer h-4 w-4 border-slate-500 text-white bg-transparent focus:ring-0 checked:bg-white checked:border-white"
                />
                <span className="text-white text-sm font-medium">JSON</span>
                <Database className="w-5 h-5 text-slate-500 ml-auto" />
              </label>
            </div>
            <button
              onClick={handleExport}
              disabled={isExporting || !messages.length}
              className="w-full mt-2 flex items-center justify-center gap-2 rounded-xl bg-primary hover:bg-blue-600 text-white font-bold h-12 transition-colors shadow-lg shadow-blue-900/20 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <Download className="w-5 h-5" />
              <span>
                {exported ? t('settings.exported') : t('settings.exportBtn')}
              </span>
            </button>
          </section>
        </div>

        <div className="p-4 border-t border-border-dark text-center">
          <p className="text-xs text-slate-600">v2.4.0 • Build 8839a</p>
        </div>
      </div>

      <ManualImportForm
        isOpen={manualOpen}
        onClose={() => setManualOpen(false)}
        partialData={manualPartial}
        missingFields={manualMissing}
        extractedTextPreview={manualPreview}
        onSubmit={handleManualSubmit}
      />
    </div>
  );
}
