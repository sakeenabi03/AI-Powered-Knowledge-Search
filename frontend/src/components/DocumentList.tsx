"use client";

import { useState } from "react";

import { formatFileSize } from "@/lib/format";
import type { DocumentItem } from "@/types";

type DocumentListProps = {
  documents: DocumentItem[];
  isLoading: boolean;
  deletingFilename: string | null;
  activeFilename?: string;
  onDelete: (filename: string) => Promise<void>;
  onAsk?: (filename: string) => void;
};

function fileTypeLabel(fileType: string): string {
  return fileType.replace(".", "").toUpperCase() || "FILE";
}

export function DocumentList({
  documents,
  isLoading,
  deletingFilename,
  activeFilename = "",
  onDelete,
  onAsk,
}: DocumentListProps) {
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Document Library
        </h2>
        <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          {isLoading ? "..." : documents.length}
        </span>
      </div>

      {isLoading ? (
        <div className="space-y-2" aria-live="polite">
          {[0, 1, 2].map((item) => (
            <div
              key={item}
              className="animate-pulse rounded-xl border border-slate-100 bg-slate-50 px-3 py-3 dark:border-slate-800 dark:bg-slate-950/60"
            >
              <div className="h-3 w-2/3 rounded bg-slate-200 dark:bg-slate-700" />
              <div className="mt-2 h-2.5 w-1/2 rounded bg-slate-200 dark:bg-slate-700" />
            </div>
          ))}
        </div>
      ) : documents.length === 0 ? (
        <p className="rounded-xl bg-slate-50 px-3 py-5 text-center text-sm text-slate-500 dark:bg-slate-950/60 dark:text-slate-400">
          No documents indexed yet.
        </p>
      ) : (
        <ul className="max-h-72 space-y-2 overflow-y-auto pr-0.5">
          {documents.map((document) => {
            const isDeleting = deletingFilename === document.filename;
            const isConfirming = pendingDelete === document.filename;
            const isActive = activeFilename === document.filename;
            const typeLabel = fileTypeLabel(document.file_type);

            return (
              <li
                key={document.filename}
                className={`rounded-xl border px-3 py-2.5 transition ${
                  isActive
                    ? "border-teal-200 bg-teal-50/70 dark:border-teal-800 dark:bg-teal-950/30"
                    : "border-slate-100 bg-slate-50/80 dark:border-slate-800 dark:bg-slate-950/50"
                } ${isDeleting ? "opacity-70" : ""}`}
              >
                <div className="flex items-start gap-2.5">
                  <div
                    className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white text-[10px] font-semibold tracking-wide text-slate-500 ring-1 ring-slate-200 dark:bg-slate-900 dark:text-slate-400 dark:ring-slate-700"
                    aria-hidden="true"
                  >
                    {typeLabel.slice(0, 4)}
                  </div>

                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {document.filename}
                    </p>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      {formatFileSize(document.file_size)} •{" "}
                      {document.indexed_chunks} passages
                    </p>
                    <span className="mt-1.5 inline-flex rounded-md bg-teal-50 px-1.5 py-0.5 text-[10px] font-medium text-teal-700 ring-1 ring-teal-100 dark:bg-teal-950/50 dark:text-teal-300 dark:ring-teal-900">
                      {isDeleting ? "Deleting..." : "Indexed"}
                    </span>
                  </div>
                </div>

                {isConfirming ? (
                  <div className="mt-2.5 rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-2 dark:border-rose-900 dark:bg-rose-950/40">
                    <p className="text-xs text-rose-700 dark:text-rose-300">
                      Delete this document and its vectors?
                    </p>
                    <div className="mt-2 flex gap-2">
                      <button
                        type="button"
                        disabled={isDeleting}
                        onClick={() => {
                          void onDelete(document.filename).finally(() => {
                            setPendingDelete(null);
                          });
                        }}
                        className="rounded-md bg-rose-700 px-2.5 py-1 text-[11px] font-medium text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 disabled:opacity-60"
                      >
                        Delete
                      </button>
                      <button
                        type="button"
                        disabled={isDeleting}
                        onClick={() => setPendingDelete(null)}
                        className="rounded-md bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 dark:bg-slate-900 dark:text-slate-200 dark:ring-slate-700"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="mt-2.5 flex items-center justify-end gap-1.5">
                    {onAsk ? (
                      <button
                        type="button"
                        disabled={isDeleting}
                        onClick={() => onAsk(document.filename)}
                        className="rounded-md bg-white px-2.5 py-1 text-[11px] font-medium text-teal-700 ring-1 ring-teal-200 transition hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 disabled:opacity-50 dark:bg-slate-900 dark:text-teal-300 dark:ring-teal-800 dark:hover:bg-teal-950/40"
                      >
                        Ask
                      </button>
                    ) : null}
                    <button
                      type="button"
                      aria-label={`Delete ${document.filename}`}
                      disabled={isDeleting || deletingFilename !== null}
                      onClick={() => setPendingDelete(document.filename)}
                      className="inline-flex h-7 w-7 items-center justify-center rounded-md text-slate-400 transition hover:bg-rose-50 hover:text-rose-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-300 disabled:opacity-50 dark:hover:bg-rose-950/40 dark:hover:text-rose-300"
                    >
                      <svg
                        viewBox="0 0 24 24"
                        className="h-3.5 w-3.5"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M6 7h12M9 7V5h6v2m-7 0v12a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V7"
                        />
                      </svg>
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
