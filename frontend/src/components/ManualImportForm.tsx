import {FormEvent, useEffect, useMemo, useRef, useState} from 'react';
import {ChevronDown, ChevronUp, X} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {useDialogFocus} from '../hooks/useDialogFocus';
import {ManualCourseData} from '../types';

type Props = {
  isOpen: boolean;
  onClose: () => void;
  partialData?: Partial<ManualCourseData>;
  missingFields?: string[];
  extractedTextPreview?: string;
  initialMessage?: string;
  onSubmit: (data: ManualCourseData) => Promise<void>;
};

const defaultForm: ManualCourseData = {
  course_code: '',
  title: '',
  term: '',
  section_id: '',
  points_raw: '',
  description: '',
  prerequisites_text: '',
};

export function parseManualPoints(raw: string): {min: number; max: number} | null {
  const match = raw.trim().match(/^(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?$/);
  if (!match) {
    return null;
  }
  const min = Number(match[1]);
  const max = Number(match[2] ?? match[1]);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min <= 0 || max > 30 || min > max) {
    return null;
  }
  return {min, max};
}

export function normalizeManualTerm(raw: string): string | null {
  const match = raw.trim().replace(/\s+/g, ' ').match(
    /^(fall|spring|summer|winter)\s+(\d{4})$/i,
  );
  if (!match) {
    return null;
  }
  return `${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()} ${match[2]}`;
}

export function isValidManualCourseCode(code: string): boolean {
  return /^[A-Z]{2,4}\s(?:[A-Z]|UN|GU|GR)\d{4}$/.test(code);
}

export default function ManualImportForm({
  isOpen,
  onClose,
  partialData,
  missingFields,
  extractedTextPreview,
  initialMessage,
  onSubmit,
}: Props) {
  const {t} = useTranslation();
  const [formData, setFormData] = useState<ManualCourseData>(defaultForm);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [invalidFields, setInvalidFields] = useState<Set<string>>(() => new Set());
  const [showPreview, setShowPreview] = useState(false);
  const firstInputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useDialogFocus({
    isOpen,
    canClose: !isSubmitting,
    onClose,
    initialFocusRef: firstInputRef,
  });

  const missingSet = useMemo(() => new Set(missingFields ?? []), [missingFields]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setError(initialMessage ?? '');
    setInvalidFields(new Set());
    setShowPreview(false);
    setFormData({
      ...defaultForm,
      ...partialData,
      course_code: partialData?.course_code ?? '',
      title: partialData?.title ?? '',
      term: partialData?.term ?? partialData?.sections?.[0]?.term ?? '',
      section_id:
        partialData?.section_id ??
        partialData?.sections?.[0]?.section_id ??
        partialData?.sections?.[0]?.section_call_number ??
        '',
      points_raw: partialData?.points_raw ?? '',
      description: partialData?.description ?? '',
      prerequisites_text: partialData?.prerequisites_text ?? '',
    });
  }, [initialMessage, isOpen, partialData]);

  if (!isOpen) {
    return null;
  }

  const isFieldInvalid = (field: string) =>
    missingSet.has(field) || invalidFields.has(field);
  const isCodeMissing = isFieldInvalid('course_code');
  const isTitleMissing = isFieldInvalid('title');

  const onLocalSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    setInvalidFields(new Set());

    // 归一化后再校验：与后端一致（大写 + 多空格压缩），
    // 这样用户输入小写/多空格也能通过，且格式要求和后端严格一致。
    const code = formData.course_code.trim().toUpperCase().replace(/\s+/g, ' ');
    const title = formData.title.trim();
    const rawTerm = formData.term.trim().replace(/\s+/g, ' ');
    const term = normalizeManualTerm(rawTerm);
    const sectionId = formData.section_id.trim().toUpperCase();

    if (!code || !title || !rawTerm || !sectionId) {
      const fields = new Set<string>();
      if (!code) fields.add('course_code');
      if (!title) fields.add('title');
      if (!rawTerm) fields.add('term');
      if (!sectionId) fields.add('section_id');
      setInvalidFields(fields);
      setError(t('settings.manualImport.required'));
      return;
    }

    // 与 shared parser 一致：任意单字母 level；双字母只允许 UN/GU/GR。
    if (!isValidManualCourseCode(code)) {
      setInvalidFields(new Set(['course_code']));
      setError(t('settings.manualImport.codeInvalid'));
      return;
    }

    if (!term) {
      setInvalidFields(new Set(['term']));
      setError(t('settings.manualImport.termInvalid'));
      return;
    }

    const points = parseManualPoints(formData.points_raw ?? '');
    if (!points) {
      setInvalidFields(new Set(['points_raw']));
      setError(
        (formData.points_raw ?? '').trim()
          ? t('settings.manualImport.pointsInvalid')
          : t('settings.manualImport.pointsRequired'),
      );
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit({
        ...formData,
        course_code: code,
        title,
        term,
        section_id: sectionId,
        points_min: points.min,
        points_max: points.max,
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
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="manual-import-title"
        tabIndex={-1}
        className="w-full max-w-2xl bg-surface-dark rounded-xl border border-border-dark shadow-2xl max-h-[90vh] overflow-hidden"
      >
        <header className="flex items-center justify-between px-5 py-4 border-b border-border-dark">
          <h3 id="manual-import-title" className="text-white text-lg font-bold">
            {t('settings.manualImport.title')}
          </h3>
          <button
            type="button"
            onClick={onClose}
            aria-label={t('settings.manualImport.close')}
            disabled={isSubmitting}
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
                aria-expanded={showPreview}
                aria-controls="manual-import-preview"
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
                <pre
                  id="manual-import-preview"
                  className="px-4 pb-4 text-xs text-slate-400 whitespace-pre-wrap break-words"
                >
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
                ref={firstInputRef}
                aria-required="true"
                aria-invalid={isCodeMissing || undefined}
                aria-describedby={error ? 'manual-import-error' : undefined}
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
                aria-required="true"
                aria-invalid={isTitleMissing || undefined}
                aria-describedby={error ? 'manual-import-error' : undefined}
                value={formData.title}
                onChange={(e) => setFormData((prev) => ({...prev, title: e.target.value}))}
                placeholder={t('settings.manualImport.courseTitlePlaceholder')}
                className={`bg-input-bg border rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary ${isTitleMissing ? 'border-red-500' : 'border-border-dark'}`}
              />
            </label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-sm text-slate-200">
                {t('settings.manualImport.term')}*
              </span>
              <input
                aria-required="true"
                aria-invalid={isFieldInvalid('term') || undefined}
                aria-describedby={error ? 'manual-import-error' : undefined}
                value={formData.term}
                onChange={(e) => setFormData((prev) => ({...prev, term: e.target.value}))}
                placeholder={t('settings.manualImport.termPlaceholder')}
                className={`bg-input-bg border rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary ${isFieldInvalid('term') ? 'border-red-500' : 'border-border-dark'}`}
              />
            </label>

            <label className="flex flex-col gap-1">
              <span className="text-sm text-slate-200">
                {t('settings.manualImport.sectionId')}*
              </span>
              <input
                aria-required="true"
                aria-invalid={
                  isFieldInvalid('section_id') ||
                  isFieldInvalid('section_call_number') ||
                  undefined
                }
                aria-describedby={error ? 'manual-import-error' : undefined}
                value={formData.section_id}
                onChange={(e) =>
                  setFormData((prev) => ({...prev, section_id: e.target.value}))
                }
                placeholder={t('settings.manualImport.sectionIdPlaceholder')}
                className={`bg-input-bg border rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary ${isFieldInvalid('section_id') || isFieldInvalid('section_call_number') ? 'border-red-500' : 'border-border-dark'}`}
              />
            </label>
          </div>

          <label className="flex flex-col gap-1">
            <span className="text-sm text-slate-200">{t('settings.manualImport.points')}*</span>
            <input
              aria-required="true"
              aria-invalid={
                isFieldInvalid('points_raw') || isFieldInvalid('points') || undefined
              }
              aria-describedby={error ? 'manual-import-error' : undefined}
              value={formData.points_raw ?? ''}
              onChange={(e) =>
                setFormData((prev) => ({...prev, points_raw: e.target.value}))
              }
              placeholder={t('settings.manualImport.pointsPlaceholder')}
              className={`bg-input-bg border rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-primary ${isFieldInvalid('points_raw') || isFieldInvalid('points') ? 'border-red-500' : 'border-border-dark'}`}
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

          {error ? (
            <p id="manual-import-error" className="text-sm text-red-400" role="alert">
              {error}
            </p>
          ) : null}

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
