import {memo, useState} from 'react';
import {Bot, Check, Copy, User} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {useTranslation} from 'react-i18next';
import {CourseSource, Message} from '../types';

type Props = {
  message: Message;
  index: number;
};

type SourceGroupProps = {
  label: string;
  sources: CourseSource[];
  group: 'answer_sources' | 'prompt_basis';
};

function SourceGroup({label, sources, group}: SourceGroupProps) {
  const {t} = useTranslation();
  const answerGroup = group === 'answer_sources';

  return (
    <section className="space-y-2 pt-1" data-source-group={group}>
      <h4 className="text-xs font-medium text-slate-400">{label}</h4>
      <div className="grid gap-2">
        {sources.map((source) => (
          <article
            key={source.uid}
            data-source-uid={source.uid}
            data-course-code={source.course_code}
            data-source-role={source.role}
            data-citation-status={source.citation_status}
            className={`min-w-0 rounded-lg border px-3 py-2 text-xs ${answerGroup ? 'border-primary/25 bg-primary/5' : 'border-border-dark bg-slate-900/30'}`}
          >
            <div className="flex min-w-0 flex-wrap items-center gap-1.5">
              <span className={answerGroup ? 'font-semibold text-primary' : 'font-semibold text-slate-300'}>
                {source.course_code}
              </span>
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">
                {source.citation_label}
              </span>
            </div>
            <p className="mt-1 break-words text-slate-300">{source.title}</p>
            <p className="mt-1 break-words text-[11px] text-slate-500">
              {source.source_label}
            </p>
            <p className="mt-1 break-all text-[10px] text-slate-600" title={source.uid}>
              {t('chat.sourceUid')}: {source.uid}
            </p>
            {source.offerings.length > 0 ? (
              <div className="mt-2 grid gap-1.5">
                {source.offerings.map((offering, offeringIndex) => (
                  <div
                    key={`${source.uid}-${offering.term ?? ''}-${offering.section_id ?? ''}-${offeringIndex}`}
                    data-source-offering-index={offeringIndex}
                    data-source-term={offering.term ?? ''}
                    data-source-section={offering.section_id ?? ''}
                    data-source-meeting-time={offering.meeting_time ?? ''}
                    data-source-location={offering.location ?? ''}
                    className="min-w-0 rounded border border-border-dark/70 bg-black/10 px-2 py-1 text-[11px] leading-relaxed text-slate-400"
                  >
                    <span>{t('chat.sourceTerm')}: {offering.term ?? t('chat.sourceUnknown')}</span>
                    <span aria-hidden="true"> · </span>
                    <span>{t('chat.sourceSection')}: {offering.section_id ?? t('chat.sourceUnknown')}</span>
                    <br />
                    <span>{t('chat.sourceMeeting')}: {offering.meeting_time ?? t('chat.sourceUnknown')}</span>
                    <span aria-hidden="true"> · </span>
                    <span>{t('chat.sourceLocation')}: {offering.location ?? t('chat.sourceUnknown')}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-[11px] text-slate-500" data-source-offerings="unknown">
                {t('chat.sourceNoOfferings')}
              </p>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function MessageBubble({message, index}: Props) {
  const [copied, setCopied] = useState(false);
  const {t} = useTranslation();
  const sources = message.sources;
  const answerSources = sources?.schema_version === 2 ? sources.answer_sources : [];
  const answerUids = new Set(answerSources.map((source) => source.uid));
  const candidateSources =
    sources?.schema_version === 2
      ? sources.prompt_basis.filter((source) => !answerUids.has(source.uid))
      : [];
  const legacyCandidates = sources?.schema_version === 1 ? sources.courses : [];
  const hasSourceDetails =
    answerSources.length > 0 || candidateSources.length > 0 || legacyCandidates.length > 0;
  const fallbackLabelKey = message.fallbackFailed
    ? message.provider === 'groq'
      ? 'chat.fallbackRestored'
      : 'chat.fallbackFailed'
    : 'chat.fallbackUsed';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      className={`flex gap-4 p-2 animate-fade-in ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
      style={{animationDelay: `${Math.min(index * 0.1, 0.5)}s`}}
    >
      <div className="flex-none">
        {message.role === 'assistant' ? (
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary ring-1 ring-primary/30">
            <Bot className="w-5 h-5" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-surface-dark flex items-center justify-center text-slate-300 ring-1 ring-border-dark">
            <User className="w-5 h-5" />
          </div>
        )}
      </div>
      <div
        className={`min-w-0 flex-1 space-y-2 flex flex-col ${message.role === 'user' ? 'items-end' : 'items-start'}`}
      >
        <div
          className={`flex flex-wrap items-baseline gap-2 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
        >
          <span className="font-bold text-sm">
            {message.role === 'assistant' ? t('chat.assistantName') : t('chat.userName')}
          </span>
          <span className="text-xs text-slate-500">{message.time}</span>
          {message.role === 'assistant' && message.provider ? (
            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-400">
              {message.provider}
            </span>
          ) : null}
          {message.role === 'assistant' && message.fallbackUsed ? (
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-medium ring-1 ${message.fallbackFailed ? 'bg-red-500/10 text-red-300 ring-red-500/20' : 'bg-amber-500/10 text-amber-300 ring-amber-500/20'}`}
              aria-label={`${t(fallbackLabelKey)}${message.fallbackReason ? `: ${message.fallbackReason}` : ''}`}
              title={message.fallbackReason || undefined}
            >
              {t(fallbackLabelKey)}
            </span>
          ) : null}
        </div>
        <div
          className={`${message.role === 'user' ? 'bg-surface-dark text-slate-100 rounded-2xl rounded-tr-sm px-5 py-3 max-w-[85%]' : 'text-slate-300 w-full'} text-[15px] leading-[1.55]`}
        >
          {message.role === 'user' ? (
            message.content
          ) : (
            <div className="space-y-2">
              <div className="min-w-0 leading-[1.65] [&_p]:my-2 [&_ul]:list-disc [&_ol]:list-decimal [&_ul]:pl-5 [&_ol]:pl-5 [&_code]:rounded [&_code]:bg-[#161b22] [&_code]:px-1.5 [&_code]:py-0.5 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:border [&_pre]:border-border-dark [&_pre]:bg-[#0d1117] [&_pre]:p-3 [&_table]:w-full [&_table]:min-w-max [&_table]:border-collapse [&_th]:border [&_th]:border-border-dark [&_th]:bg-[#161b22] [&_th]:px-3 [&_th]:py-2 [&_td]:border [&_td]:border-border-dark [&_td]:px-3 [&_td]:py-2">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({children, ...props}) => (
                      <div className="my-3 max-w-full overflow-x-auto rounded-lg">
                        <table {...props}>{children}</table>
                      </div>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
                {message.isStreaming ? (
                  <span className="inline-block ml-1 animate-pulse text-primary">▍</span>
                ) : null}
              </div>
              {!message.isStreaming && message.status && message.status !== 'complete' ? (
                <p
                  className={`text-xs ${message.status === 'stopped' ? 'text-amber-300' : 'text-red-300'}`}
                  role="status"
                >
                  {t(`chat.status.${message.status}`)}
                </p>
              ) : null}
              {!message.isStreaming && hasSourceDetails ? (
                <div className="w-full space-y-2" data-source-schema-version={sources?.schema_version}>
                  {answerSources.length > 0 ? (
                    <SourceGroup
                      label={t('chat.verifiedSources')}
                      sources={answerSources}
                      group="answer_sources"
                    />
                  ) : null}
                  {candidateSources.length > 0 ? (
                    <SourceGroup
                      label={t('chat.candidateSources')}
                      sources={candidateSources}
                      group="prompt_basis"
                    />
                  ) : null}
                  {legacyCandidates.length > 0 ? (
                    <section className="space-y-2 pt-1" data-source-group="prompt_basis">
                      <h4 className="text-xs font-medium text-slate-400">
                        {t('chat.candidateSources')}
                      </h4>
                      <div className="flex flex-wrap gap-1.5">
                        {legacyCandidates.map((code, sourceIndex) => (
                          <span
                            key={`${code}-${sourceIndex}`}
                            data-source-legacy="true"
                            data-course-code={code}
                            className="rounded-md bg-slate-800 px-2 py-0.5 text-xs font-medium text-slate-300 ring-1 ring-border-dark"
                          >
                            {code}
                          </span>
                        ))}
                      </div>
                    </section>
                  ) : null}
                </div>
              ) : null}
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={handleCopy}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                  aria-label={copied ? t('chat.copied') : t('chat.copy')}
                  aria-live="polite"
                >
                  {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? t('chat.copied') : t('chat.copy')}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(MessageBubble);
