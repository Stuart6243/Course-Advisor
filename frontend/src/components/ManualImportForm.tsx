import {FormEvent, useEffect, useMemo, useState} from 'react';
import {ChevronDown, ChevronUp, X} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {ManualCourseData} from '../types';

type Props = {
  isOpen: boolean;
  onClose: () => void;
  partialData?: Partial<ManualCourseData>;
  missingFields?: string[];
  extractedTextPreview?: string;
  onSubmit: (data: ManualCourseData) => Promise<void>;
};

const defaultForm: ManualCourseData = {
  course_code: '',
  title: '',
  points_raw: '',
  description: '',
  prerequisites_text: '',
};

export default function ManualImportForm({
  isOpen,
  onClose,
  partialData,
  missingFields,
  extractedTextPreview,
  onSubmit,
}: Props) {
  const {t} = useTranslation();
  const [formData, setFormData] = useState<ManualCourseData>(defaultForm);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [showPreview, setShowPreview] = useState(false);

  const missingSet = useMemo(() => new Set(missingFields ?? []), [missingFields]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setError('');
    setShowPreview(false);
    setFormData({
      ...defaultForm,
      ...partialData,
      course_code: partialData?.course_code ?? '',
      title: partialData?.title ?? '',
      points_raw: partialData?.points_raw ?? '',
      description: partialData?.description ?? '',
      prerequisites_text: partialData?.prerequisites_text ?? '',
    });
  }, [isOpen, partialData]);

  if (!isOpen) {
    return null;
  }

  const isCodeMissing = missingSet.has('course_code');
  const isTitleMissing = missingSet.has('title');

  const onLocalSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');

    const code = formData.course_code.trim();
    const title = formData.title.trim();

    if (!code || !title) {
      setError(t('settings.manualImport.required'));
      return;
    }

    if (!/(?=.*[A-Z])(?=.*\d)/.test(code)) {
      setError(t('settings.manualImport.codeInvalid'));
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit({
        ...formData,
        course_code: code,
        title,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : t('settings.importError');
      setError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/65 px-4">
      <div className="w-full max-w-2xl bg-surface-dark rounded-xl border border-border-dark shadow-2xl max-h-[90vh] overflow-hidden">
        <header className="flex items-center justify-between px-5 py-4 border-b border-border-dark">
          <h3 className="text-white text-lg font-bold">{t('settings.manualImport.title')}</h3>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors rounded-full p-1"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        <form onSubmit={onLocalSubmit} className="p-5 space-y-4 overflow-y-auto max-h-[calc(90vh-70px)]">
          <p className="text-slate-300 text-sm">{t('settings.manualImport.description')}</p>

          {extractedTextPreview ? (
            <div className="rounded-lg border border-border-dark bg-input-bg/50">
              <button
                type="button"
                onClick={() => setShowPreview((v) => !v)}
                className="w-full px-4 py-3 text-left text-sm text-slate-200 flex items-center justify-between"
              >
                <span>{t('settings.manualImport.extractedPreview')}</span>
                {showPreview ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
              {showPreview ? (
                <pre className="px-4 pb-4 text-xs text-slate-400 whitespace-pre-wrap break-words">
                  {extractedTextPreview}
                </pre>
              ) : null}
            </div>
          ) : null}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-sm text-slate-200">
                {t('settings.manualImport.courseCode')}*
              </span>
              <input
                value={formData.course_code}
                onChange={(e) =>
                  setFormData((prev) => ({...prev, course_code: e.target.value}))
                }
                placeholder={t('settings.manualImport.courseCodePlaceholder')}
                className={`bg-input-bg border rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary ${isCodeMissing ? 'border-red-500' : 'border-border-dark'}`}
              />
            </label>

            <label className="flex flex-col gap-1">
              <span className="text-sm text-slate-200">
                {t('settings.manualImport.courseTitle')}*
              </span>
              <input
                value={formData.title}
                onChange={(e) => setFormData((prev) => ({...prev, title: e.target.value}))}
                placeholder={t('settings.manualImport.courseTitlePlaceholder')}
                className={`bg-input-bg border rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary ${isTitleMissing ? 'border-red-500' : 'border-border-dark'}`}
              />
            </label>
          </div>

          <label className="flex flex-col gap-1">
            <span className="text-sm text-slate-200">{t('settings.manualImport.points')}</span>
            <input
              value={formData.points_raw ?? ''}
              onChange={(e) =>
                setFormData((prev) => ({...prev, points_raw: e.target.value}))
              }
              placeholder={t('settings.manualImport.pointsPlaceholder')}
              className="bg-input-bg border border-border-dark rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm text-slate-200">{t('settings.manualImport.description')}</span>
            <textarea
              value={formData.description ?? ''}
              onChange={(e) =>
                setFormData((prev) => ({...prev, description: e.target.value}))
              }
              className="bg-input-bg border border-border-dark rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary min-h-24"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm text-slate-200">
              {t('settings.manualImport.prerequisites')}
            </span>
            <textarea
              value={formData.prerequisites_text ?? ''}
              onChange={(e) =>
                setFormData((prev) => ({...prev, prerequisites_text: e.target.value}))
              }
              className="bg-input-bg border border-border-dark rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary min-h-24"
            />
          </label>

          {error ? <p className="text-sm text-red-400">{error}</p> : null}

          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="border border-border-dark text-slate-400 rounded-lg px-4 py-2 hover:text-white hover:border-slate-500 transition-colors"
              disabled={isSubmitting}
            >
              {t('settings.manualImport.cancel')}
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="bg-primary hover:bg-primary/80 text-white rounded-lg px-4 py-2 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {isSubmitting ? t('settings.importProcessing') : t('settings.manualImport.submit')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
