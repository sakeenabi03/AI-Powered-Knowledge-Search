"use client";

import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";

type UploadCardProps = {
  acceptedFormats?: string;
  isUploading: boolean;
  onUpload: (file: File) => Promise<void>;
};

const ACCEPTED_EXTENSIONS = new Set([".pdf", ".docx", ".txt"]);
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

const UPLOAD_STATUS_MESSAGES = [
  "Uploading document...",
  "Processing text...",
  "Creating embeddings...",
  "Indexing vectors...",
];

function getExtension(filename: string): string {
  const index = filename.lastIndexOf(".");
  if (index < 0) {
    return "";
  }
  return filename.slice(index).toLowerCase();
}

export function UploadCard({
  acceptedFormats = "PDF, DOCX, TXT",
  isUploading,
  onUpload,
}: UploadCardProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [statusIndex, setStatusIndex] = useState(0);

  useEffect(() => {
    if (!isUploading) {
      return;
    }

    const timer = window.setInterval(() => {
      setStatusIndex((previous) => (previous + 1) % UPLOAD_STATUS_MESSAGES.length);
    }, 1800);

    return () => window.clearInterval(timer);
  }, [isUploading]);

  const processFile = async (file: File | undefined) => {
    if (!file || isUploading) {
      return;
    }

    const extension = getExtension(file.name);
    if (!ACCEPTED_EXTENSIONS.has(extension)) {
      setLocalError("Unsupported file type. Only PDF, DOCX, and TXT are allowed.");
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setLocalError("File exceeds the 10 MB size limit.");
      return;
    }

    setLocalError(null);
    setStatusIndex(0);
    try {
      await onUpload(file);
    } catch {
      // Parent component surfaces API errors via inline feedback.
    } finally {
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    void processFile(file);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    void processFile(file);
  };

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div
        onDragEnter={(event) => {
          event.preventDefault();
          if (!isUploading) {
            setIsDragging(true);
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={handleDrop}
        className={`flex min-h-[140px] flex-col items-center justify-center rounded-xl border border-dashed px-4 py-4 text-center transition ${
          isDragging
            ? "border-teal-500 bg-teal-50 dark:border-teal-400 dark:bg-teal-950/40"
            : "border-slate-200 bg-slate-50/80 dark:border-slate-700 dark:bg-slate-950/50"
        } ${isUploading ? "opacity-70" : ""}`}
      >
        <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-teal-50 text-teal-700 dark:bg-teal-950/60 dark:text-teal-300">
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
              d="M12 16V4m0 0 4 4m-4-4-4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
            />
          </svg>
        </div>

        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          {isUploading ? UPLOAD_STATUS_MESSAGES[statusIndex] : "+ Upload Document"}
        </p>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          {acceptedFormats} • Max 10 MB
        </p>
        {!isUploading ? (
          <p className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">
            Drag and drop or choose a file
          </p>
        ) : (
          <div
            className="mt-2 flex items-center gap-1 text-teal-700 dark:text-teal-300"
            aria-live="polite"
          >
            <span className="animate-dot h-1 w-1 rounded-full bg-current" />
            <span className="animate-dot animate-dot-delay-1 h-1 w-1 rounded-full bg-current" />
            <span className="animate-dot animate-dot-delay-2 h-1 w-1 rounded-full bg-current" />
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          className="hidden"
          disabled={isUploading}
          onChange={handleInputChange}
        />
        <button
          type="button"
          disabled={isUploading}
          onClick={() => inputRef.current?.click()}
          className="mt-3 rounded-lg bg-slate-900 px-3.5 py-1.5 text-xs font-medium text-white transition hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 disabled:opacity-60 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white dark:focus-visible:ring-offset-slate-900"
        >
          {isUploading ? "Processing..." : "Choose File"}
        </button>
      </div>

      {localError ? (
        <p
          className="mt-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-300"
          role="alert"
        >
          {localError}
        </p>
      ) : null}
    </section>
  );
}
