"use client";

import { useState, type KeyboardEvent, type MouseEvent } from "react";

import type { ChatSource } from "@/types";

type SourceCardProps = {
  source: ChatSource;
  isSelected?: boolean;
  onSelect?: (chunkId: string) => void;
};

export function SourceCard({
  source,
  isSelected = false,
  onSelect,
}: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const similarityPercent = Math.round(source.similarity_score * 100);
  const fullContent = source.content ?? "";
  const preview = source.excerpt ?? "";
  const canToggle =
    fullContent.trim().length > 0 &&
    fullContent !== preview &&
    fullContent.length > preview.length;
  const visibleText = expanded && canToggle ? fullContent : preview;

  const handleSelect = () => {
    onSelect?.(source.chunk_id);
  };

  const handleCardKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleSelect();
    }
  };

  const handleToggle = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setExpanded((previous) => !previous);
  };

  return (
    <article
      role="button"
      tabIndex={0}
      aria-pressed={isSelected}
      onClick={handleSelect}
      onKeyDown={handleCardKeyDown}
      className={`cursor-pointer rounded-xl border px-3.5 py-3.5 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 ${
        isSelected
          ? "border-teal-300 bg-teal-50/70 shadow-sm dark:border-teal-700 dark:bg-teal-950/40"
          : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-slate-600 dark:hover:bg-slate-800/80"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Reference {source.source_number}
        </p>
        <p className="shrink-0 rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] font-medium text-slate-500 ring-1 ring-slate-200/80 dark:bg-slate-800 dark:text-slate-400 dark:ring-slate-700">
          {similarityPercent}% match
        </p>
      </div>

      <div className="mt-3">
        <p className="text-[10px] font-medium tracking-wide text-slate-400 uppercase">
          Document
        </p>
        <p className="mt-0.5 truncate text-xs text-slate-500 dark:text-slate-400">
          {source.filename}
        </p>
      </div>

      <div className="mt-2.5">
        <p className="text-[10px] font-medium tracking-wide text-slate-400 uppercase">
          Retrieved Passage
        </p>
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
          Passage {source.chunk_index}
        </p>
      </div>

      <div className="mt-3">
        <p className="text-[10px] font-medium tracking-wide text-slate-400 uppercase">
          Excerpt
        </p>
        <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700 dark:text-slate-200">
          {visibleText}
        </p>
      </div>

      {canToggle ? (
        <button
          type="button"
          aria-expanded={expanded}
          className="mt-2.5 text-[11px] font-medium text-slate-600 underline-offset-2 transition hover:text-slate-900 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-slate-300 dark:hover:text-slate-100"
          onClick={handleToggle}
          onKeyDown={(event) => {
            event.stopPropagation();
          }}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </article>
  );
}
