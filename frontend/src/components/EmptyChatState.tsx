type EmptyChatStateProps = {
  onSelectQuestion: (question: string) => void;
  disabled?: boolean;
};

const SUGGESTED_QUESTIONS = [
  "Summarize this document",
  "What are the main objectives?",
  "What are the key deliverables?",
  "Why was the GRU model used?",
];

export function EmptyChatState({
  onSelectQuestion,
  disabled = false,
}: EmptyChatStateProps) {
  return (
    <div className="flex flex-col items-center justify-center px-4 py-8 text-center sm:py-10">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-teal-50 text-teal-700 ring-1 ring-teal-100 dark:bg-teal-950/50 dark:text-teal-300 dark:ring-teal-900">
        <svg
          viewBox="0 0 24 24"
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M8 10h8M8 14h5M7 4h10a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-4l-4 3v-3H7a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z"
          />
        </svg>
      </div>

      <h3 className="mt-4 text-base font-semibold text-slate-900 dark:text-slate-100">
        Ask your documents
      </h3>
      <p className="mt-1.5 max-w-md text-sm text-slate-500 dark:text-slate-400">
        Search across indexed knowledge and receive answers with source
        citations.
      </p>

      <div className="mt-5 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTED_QUESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            disabled={disabled}
            onClick={() => onSelectQuestion(question)}
            className="rounded-xl border border-slate-200 bg-white px-3.5 py-3 text-left text-sm text-slate-700 shadow-sm transition hover:border-teal-200 hover:bg-teal-50/50 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-teal-800 dark:hover:bg-teal-950/30 dark:hover:text-slate-50"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}
