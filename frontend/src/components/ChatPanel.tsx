"use client";

import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";

import { EmptyChatState } from "@/components/EmptyChatState";
import { SourceCard } from "@/components/SourceCard";
import { ThemeToggle } from "@/components/ThemeToggle";
import { askQuestion } from "@/lib/api";
import { formatModelName } from "@/lib/format";
import type { ChatMessage, DocumentItem, DocumentStats } from "@/types";

type ChatPanelProps = {
  documents: DocumentItem[];
  stats: DocumentStats;
  selectedFilename: string;
  onSelectedFilenameChange: (filename: string) => void;
  placeholderQuestion?: string;
};

const TOP_K_OPTIONS = [3, 5, 8] as const;

function createMessageId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function LoadingDots() {
  return (
    <span className="ml-1 inline-flex items-center gap-0.5" aria-hidden="true">
      <span className="animate-dot h-1 w-1 rounded-full bg-current" />
      <span className="animate-dot animate-dot-delay-1 h-1 w-1 rounded-full bg-current" />
      <span className="animate-dot animate-dot-delay-2 h-1 w-1 rounded-full bg-current" />
    </span>
  );
}

export function ChatPanel({
  documents,
  stats,
  selectedFilename,
  onSelectedFilenameChange,
  placeholderQuestion = "Ask anything about your indexed documents...",
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState<(typeof TOP_K_OPTIONS)[number]>(5);
  const [isSending, setIsSending] = useState(false);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [lastFailedQuestion, setLastFailedQuestion] = useState<string | null>(
    null,
  );
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  const activeFilename = documents.some(
    (document) => document.filename === selectedFilename,
  )
    ? selectedFilename
    : "";

  const submitQuestion = async (rawQuestion: string) => {
    const cleaned = rawQuestion.trim();
    if (!cleaned || isSending) {
      return;
    }

    const userMessage: ChatMessage = {
      id: createMessageId(),
      role: "user",
      content: cleaned,
      createdAt: new Date().toISOString(),
    };

    setMessages((previous) => [...previous, userMessage]);
    setQuestion("");
    setIsSending(true);
    setSelectedSourceId(null);
    setLastFailedQuestion(null);

    try {
      const response = await askQuestion(
        cleaned,
        topK,
        activeFilename || null,
      );
      setMessages((previous) => [
        ...previous,
        {
          id: createMessageId(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          model: response.model,
          createdAt: new Date().toISOString(),
          retrievedCount: response.retrieved_count,
        },
      ]);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to generate an answer.";
      setLastFailedQuestion(cleaned);
      setMessages((previous) => [
        ...previous,
        {
          id: createMessageId(),
          role: "assistant",
          content: message,
          createdAt: new Date().toISOString(),
          isError: true,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleSubmit = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    await submitQuestion(question);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  };

  const handleClearChat = () => {
    if (isSending) {
      return;
    }
    setMessages([]);
    setSelectedSourceId(null);
    setLastFailedQuestion(null);
  };

  const handleRetry = () => {
    if (!lastFailedQuestion || isSending) {
      return;
    }
    setMessages((previous) => previous.filter((message) => !message.isError));
    void submitQuestion(lastFailedQuestion);
  };

  const latestModel = [...messages]
    .reverse()
    .find((message) => message.role === "assistant" && message.model)?.model;
  const modelLabel = formatModelName(latestModel);

  const kpiCards = [
    {
      label: "Documents",
      value: String(stats.total_documents),
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M7 4h7l3 3v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z"
        />
      ),
    },
    {
      label: "Indexed Passages",
      value: String(stats.total_vectors),
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M4 7h16M4 12h10M4 17h7"
        />
      ),
    },
    {
      label: "Knowledge Base",
      value: stats.total_documents > 0 ? "Ready" : "Idle",
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 3l7 4v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V7l7-4Z"
        />
      ),
    },
  ] as const;

  const selectClassName =
    "rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-sm text-slate-800 outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-teal-400 dark:focus:ring-teal-900";

  return (
    <section className="flex h-full min-h-[36rem] flex-col lg:min-h-[calc(100vh-2rem)]">
      <header className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl dark:text-slate-50">
            Ask your knowledge base
          </h2>
          <p className="mt-1 max-w-xl text-sm text-slate-500 dark:text-slate-400">
            Search, summarize, and analyze your indexed AI-Powered Knowledge Search documents.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 dark:bg-slate-900 dark:text-slate-300 dark:ring-slate-700">
            {stats.total_documents}{" "}
            {stats.total_documents === 1 ? "Document" : "Documents"}
          </span>
          <span className="rounded-full bg-white px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 dark:bg-slate-900 dark:text-slate-300 dark:ring-slate-700">
            {stats.total_vectors} Passages
          </span>
          <ThemeToggle />
        </div>
      </header>

      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        {kpiCards.map((card) => (
          <article
            key={card.label}
            className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm dark:border-slate-800 dark:bg-slate-900"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] font-medium tracking-wide text-slate-400 uppercase">
                {card.label}
              </p>
              <svg
                viewBox="0 0 24 24"
                className="h-4 w-4 text-slate-300 dark:text-slate-600"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                aria-hidden="true"
              >
                {card.icon}
              </svg>
            </div>
            <p className="mt-1 text-xl font-semibold text-slate-900 dark:text-slate-50">
              {card.value}
            </p>
          </article>
        ))}
      </div>

      {modelLabel ? (
        <div className="-mt-2 mb-4">
          <span className="inline-flex rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
            Model · {modelLabel}
          </span>
        </div>
      ) : null}

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <div className="flex flex-col gap-3 border-b border-slate-100 px-4 py-3 sm:flex-row sm:items-center sm:justify-between dark:border-slate-800">
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-xs font-medium text-slate-500 dark:text-slate-400">
              Scope
              <select
                value={activeFilename}
                disabled={isSending}
                onChange={(event) =>
                  onSelectedFilenameChange(event.target.value)
                }
                className={`max-w-[220px] ${selectClassName}`}
              >
                <option value="">All Documents</option>
                {documents.map((document) => (
                  <option key={document.filename} value={document.filename}>
                    {document.filename}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex items-center gap-2 text-xs font-medium text-slate-500 dark:text-slate-400">
              Retrieval
              <select
                value={topK}
                disabled={isSending}
                onChange={(event) =>
                  setTopK(
                    Number(event.target.value) as (typeof TOP_K_OPTIONS)[number],
                  )
                }
                className={selectClassName}
              >
                {TOP_K_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    Top {option}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <button
            type="button"
            disabled={isSending || messages.length === 0}
            onClick={handleClearChat}
            className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-500 transition hover:bg-slate-50 hover:text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 disabled:opacity-40 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200 dark:focus-visible:ring-slate-600"
          >
            Clear Chat
          </button>
        </div>

        {activeFilename ? (
          <div className="flex items-center justify-between gap-3 border-b border-teal-100 bg-teal-50/70 px-4 py-2.5 dark:border-teal-900 dark:bg-teal-950/40">
            <p className="min-w-0 truncate text-xs text-teal-800 dark:text-teal-200">
              <span className="font-medium">Searching only:</span>{" "}
              {activeFilename}
            </p>
            <button
              type="button"
              aria-label="Clear document filter"
              disabled={isSending}
              onClick={() => onSelectedFilenameChange("")}
              className="shrink-0 rounded-md px-2 py-1 text-xs font-medium text-teal-700 transition hover:bg-teal-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 disabled:opacity-50 dark:text-teal-300 dark:hover:bg-teal-900/50"
            >
              Clear filter
            </button>
          </div>
        ) : null}

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
          {messages.length === 0 && !isSending ? (
            <EmptyChatState
              disabled={isSending}
              onSelectQuestion={(suggested) => {
                void submitQuestion(suggested);
              }}
            />
          ) : (
            messages.map((message) => {
              const isError = Boolean(message.isError);
              const retrievedCount =
                message.retrievedCount ?? message.sources?.length ?? 0;
              const messageModel = formatModelName(message.model);

              if (message.role === "user") {
                return (
                  <div key={message.id} className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl bg-slate-900 px-4 py-2.5 text-sm leading-relaxed text-white sm:max-w-[70%] dark:bg-slate-100 dark:text-slate-900">
                      <p className="whitespace-pre-wrap break-words">
                        {message.content}
                      </p>
                    </div>
                  </div>
                );
              }

              if (isError) {
                return (
                  <div key={message.id} className="flex justify-start">
                    <div className="w-full max-w-2xl rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 dark:border-rose-900 dark:bg-rose-950/40">
                      <p className="text-sm font-medium text-rose-800 dark:text-rose-200">
                        Unable to generate an answer.
                      </p>
                      <p className="mt-1 text-xs text-rose-700 dark:text-rose-300">
                        Please check that the API is running and try again.
                      </p>
                      {message.content ? (
                        <p className="mt-2 text-xs text-rose-600/90 dark:text-rose-300/90">
                          {message.content}
                        </p>
                      ) : null}
                      <button
                        type="button"
                        disabled={isSending || !lastFailedQuestion}
                        onClick={handleRetry}
                        className="mt-3 rounded-lg bg-rose-700 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-rose-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 disabled:opacity-50"
                      >
                        Retry
                      </button>
                    </div>
                  </div>
                );
              }

              return (
                <div key={message.id} className="flex justify-start">
                  <article className="w-full max-w-4xl rounded-2xl border border-slate-200 bg-slate-50/60 shadow-sm dark:border-slate-800 dark:bg-slate-950/50">
                    <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/80 px-4 py-3 dark:border-slate-800">
                      <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                        Grounded Answer
                      </h3>
                      <div className="flex flex-wrap items-center gap-1.5">
                        {messageModel ? (
                          <span className="rounded-md bg-white px-2 py-0.5 text-[11px] font-medium text-slate-700 ring-1 ring-slate-200 dark:bg-slate-900 dark:text-slate-200 dark:ring-slate-700">
                            {messageModel}
                          </span>
                        ) : null}
                        <span className="rounded-md bg-white px-2 py-0.5 text-[11px] font-medium text-slate-700 ring-1 ring-slate-200 dark:bg-slate-900 dark:text-slate-200 dark:ring-slate-700">
                          {retrievedCount}{" "}
                          {retrievedCount === 1 ? "reference" : "references"}
                        </span>
                        <span className="max-w-[180px] truncate rounded-md bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-400 ring-1 ring-slate-100 dark:bg-slate-900/80 dark:text-slate-500 dark:ring-slate-800">
                          {activeFilename || "All documents"}
                        </span>
                      </div>
                    </header>

                    <div className="px-4 py-3">
                      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-800 dark:text-slate-200">
                        {message.content}
                      </p>
                    </div>

                    {message.sources && message.sources.length > 0 ? (
                      <footer className="border-t border-slate-200/80 px-4 py-3 dark:border-slate-800">
                        <p className="mb-2 text-xs font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
                          References
                        </p>
                        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                          {message.sources.map((source) => (
                            <SourceCard
                              key={`${message.id}-${source.chunk_id}`}
                              source={source}
                              isSelected={
                                selectedSourceId === source.chunk_id
                              }
                              onSelect={setSelectedSourceId}
                            />
                          ))}
                        </div>
                      </footer>
                    ) : null}
                  </article>
                </div>
              );
            })
          )}

          {isSending ? (
            <div className="flex justify-start">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-300">
                Analyzing relevant document sources
                <LoadingDots />
              </div>
            </div>
          ) : null}
          <div ref={messagesEndRef} />
        </div>

        <form
          className="border-t border-slate-100 px-4 py-3 dark:border-slate-800"
          onSubmit={(event) => {
            void handleSubmit(event);
          }}
        >
          <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-2 shadow-sm focus-within:border-teal-300 focus-within:ring-2 focus-within:ring-teal-100 dark:border-slate-700 dark:bg-slate-950/50 dark:focus-within:border-teal-700 dark:focus-within:ring-teal-950">
            <label htmlFor="question" className="sr-only">
              Question
            </label>
            <textarea
              id="question"
              rows={2}
              value={question}
              disabled={isSending}
              placeholder={placeholderQuestion}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full resize-none bg-transparent px-2 py-2 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60 dark:text-slate-100 dark:placeholder:text-slate-500"
            />
            <div className="flex flex-col gap-2 px-1 pb-1 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-[11px] text-slate-400 dark:text-slate-500">
                {isSending
                  ? "Searching documents..."
                  : "Enter to send • Shift + Enter for new line"}
              </p>
              <button
                type="submit"
                disabled={isSending || question.trim().length === 0}
                className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-teal-700 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-teal-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 disabled:opacity-60 dark:bg-teal-600 dark:hover:bg-teal-500 dark:focus-visible:ring-offset-slate-900"
              >
                <svg
                  viewBox="0 0 24 24"
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5 12h14m0 0-5-5m5 5-5 5"
                  />
                </svg>
                {isSending ? "Searching..." : "Ask"}
              </button>
            </div>
          </div>
        </form>
      </div>
    </section>
  );
}
