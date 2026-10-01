import type { DocumentItem } from "@/types";

import { DocumentList } from "@/components/DocumentList";
import { StatsCards } from "@/components/StatsCards";
import { UploadCard } from "@/components/UploadCard";

type SidebarProps = {
  documents: DocumentItem[];
  totalDocuments: number;
  totalVectors: number;
  collectionName: string;
  isLoading: boolean;
  isUploading: boolean;
  deletingFilename: string | null;
  activeFilename?: string;
  systemReady?: boolean;
  onUpload: (file: File) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  onAskDocument?: (filename: string) => void;
};

export function Sidebar({
  documents,
  totalDocuments,
  totalVectors,
  collectionName,
  isLoading,
  isUploading,
  deletingFilename,
  activeFilename = "",
  systemReady = true,
  onUpload,
  onDelete,
  onAskDocument,
}: SidebarProps) {
  return (
    <aside className="flex h-full flex-col gap-3 lg:max-h-screen lg:overflow-y-auto lg:pr-1">
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-900 text-xs font-semibold tracking-wide text-white dark:bg-slate-100 dark:text-slate-900">
            KDA
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
              AI-Powered Knowledge Search
            </h1>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Enterprise RAG Workspace
            </p>
          </div>
        </div>

        <div className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-teal-50 px-2.5 py-1 text-[11px] font-medium text-teal-700 ring-1 ring-teal-100 dark:bg-teal-950/50 dark:text-teal-300 dark:ring-teal-900">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              systemReady ? "bg-emerald-500" : "bg-amber-500"
            }`}
            aria-hidden="true"
          />
          {systemReady ? "System Ready" : "Reconnecting..."}
        </div>
      </div>

      <UploadCard isUploading={isUploading} onUpload={onUpload} />

      <StatsCards
        totalDocuments={totalDocuments}
        totalVectors={totalVectors}
        collectionName={collectionName}
        isLoading={isLoading}
      />

      <DocumentList
        documents={documents}
        isLoading={isLoading}
        deletingFilename={deletingFilename}
        activeFilename={activeFilename}
        onDelete={onDelete}
        onAsk={onAskDocument}
      />
    </aside>
  );
}
