"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  deleteDocument,
  getDocuments,
  getStats,
  uploadDocument,
} from "@/lib/api";
import type { DocumentItem, DocumentStats } from "@/types";

import { Sidebar } from "@/components/Sidebar";

type Feedback = {
  type: "success" | "error";
  message: string;
};

type DocumentsWorkspaceProps = {
  onDocumentsChange?: (documents: DocumentItem[]) => void;
  onStatsChange?: (stats: DocumentStats) => void;
  activeFilename?: string;
  onAskDocument?: (filename: string) => void;
};

const EMPTY_STATS: DocumentStats = {
  total_documents: 0,
  total_vectors: 0,
  collection_name: "-",
};

export function DocumentsWorkspace({
  onDocumentsChange,
  onStatsChange,
  activeFilename = "",
  onAskDocument,
}: DocumentsWorkspaceProps) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [stats, setStats] = useState<DocumentStats>(EMPTY_STATS);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const feedbackTimerRef = useRef<number | null>(null);

  const showFeedback = useCallback((next: Feedback) => {
    setFeedback(next);
    if (feedbackTimerRef.current !== null) {
      window.clearTimeout(feedbackTimerRef.current);
    }
    feedbackTimerRef.current = window.setTimeout(() => {
      setFeedback(null);
      feedbackTimerRef.current = null;
    }, 4000);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function fetchWorkspaceData() {
      try {
        const [nextDocuments, nextStats] = await Promise.all([
          getDocuments(),
          getStats(),
        ]);
        if (cancelled) {
          return;
        }
        setDocuments(nextDocuments);
        onDocumentsChange?.(nextDocuments);
        setStats(nextStats);
        onStatsChange?.(nextStats);
        setConnectionError(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        const message =
          error instanceof Error
            ? error.message
            : "Unable to connect to the backend service.";
        setConnectionError(message);
        setDocuments([]);
        onDocumentsChange?.([]);
        setStats(EMPTY_STATS);
        onStatsChange?.(EMPTY_STATS);
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void fetchWorkspaceData();

    return () => {
      cancelled = true;
      if (feedbackTimerRef.current !== null) {
        window.clearTimeout(feedbackTimerRef.current);
      }
    };
  }, [onDocumentsChange, onStatsChange, reloadToken]);

  const handleRetry = () => {
    setIsLoading(true);
    setConnectionError(null);
    setReloadToken((token) => token + 1);
  };

  const handleUpload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setFeedback(null);

      try {
        const result = await uploadDocument(file);
        const [nextDocuments, nextStats] = await Promise.all([
          getDocuments(),
          getStats(),
        ]);
        setDocuments(nextDocuments);
        onDocumentsChange?.(nextDocuments);
        setStats(nextStats);
        onStatsChange?.(nextStats);
        setConnectionError(null);
        showFeedback({
          type: "success",
          message: result.message || "Document uploaded and indexed successfully.",
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to upload document.";
        showFeedback({ type: "error", message });
        throw error;
      } finally {
        setIsUploading(false);
      }
    },
    [onDocumentsChange, onStatsChange, showFeedback],
  );

  const handleDelete = useCallback(
    async (filename: string) => {
      setDeletingFilename(filename);
      setFeedback(null);

      try {
        const result = await deleteDocument(filename);
        const [nextDocuments, nextStats] = await Promise.all([
          getDocuments(),
          getStats(),
        ]);
        setDocuments(nextDocuments);
        onDocumentsChange?.(nextDocuments);
        setStats(nextStats);
        onStatsChange?.(nextStats);
        setConnectionError(null);
        showFeedback({
          type: "success",
          message:
            result.message || "Document and vector records deleted successfully.",
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to delete document.";
        showFeedback({ type: "error", message });
      } finally {
        setDeletingFilename(null);
      }
    },
    [onDocumentsChange, onStatsChange, showFeedback],
  );

  return (
    <div className="flex flex-col gap-3">
      {feedback ? (
        <div
          className={`rounded-xl px-3 py-2 text-sm ${
            feedback.type === "success"
              ? "border border-teal-200 bg-teal-50 text-teal-800 dark:border-teal-900 dark:bg-teal-950/40 dark:text-teal-200"
              : "border border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-300"
          }`}
          role="status"
        >
          {feedback.message}
        </div>
      ) : null}

      {connectionError ? (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 shadow-sm dark:border-rose-900 dark:bg-rose-950/40">
          <p className="text-sm font-medium text-rose-800 dark:text-rose-200">
            Unable to reach the API
          </p>
          <p className="mt-1 text-xs text-rose-700 dark:text-rose-300">
            {connectionError}
          </p>
          <button
            type="button"
            onClick={handleRetry}
            className="mt-3 rounded-lg bg-rose-700 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-rose-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
          >
            Retry
          </button>
        </div>
      ) : null}

      <Sidebar
        documents={documents}
        totalDocuments={stats.total_documents}
        totalVectors={stats.total_vectors}
        collectionName={stats.collection_name}
        isLoading={isLoading}
        isUploading={isUploading}
        deletingFilename={deletingFilename}
        activeFilename={activeFilename}
        systemReady={!connectionError}
        onUpload={handleUpload}
        onDelete={handleDelete}
        onAskDocument={onAskDocument}
      />
    </div>
  );
}
