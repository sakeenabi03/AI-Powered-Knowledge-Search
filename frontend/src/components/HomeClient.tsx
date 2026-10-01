"use client";

import { useCallback, useState } from "react";

import { ChatPanel } from "@/components/ChatPanel";
import { DocumentsWorkspace } from "@/components/DocumentsWorkspace";
import type { DocumentItem, DocumentStats } from "@/types";

const EMPTY_STATS: DocumentStats = {
  total_documents: 0,
  total_vectors: 0,
  collection_name: "-",
};

export function HomeClient() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [stats, setStats] = useState<DocumentStats>(EMPTY_STATS);
  const [selectedFilename, setSelectedFilename] = useState("");

  const handleDocumentsChange = useCallback((nextDocuments: DocumentItem[]) => {
    setDocuments(nextDocuments);
    setSelectedFilename((current) =>
      current && nextDocuments.some((document) => document.filename === current)
        ? current
        : "",
    );
  }, []);

  const handleStatsChange = useCallback((nextStats: DocumentStats) => {
    setStats(nextStats);
  }, []);

  const handleAskDocument = useCallback((filename: string) => {
    setSelectedFilename(filename);
  }, []);

  return (
    <div className="mx-auto grid min-h-screen w-full max-w-[1600px] gap-4 px-3 py-3 sm:px-4 lg:grid-cols-[320px_minmax(0,1fr)] lg:gap-5 lg:px-5 lg:py-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <DocumentsWorkspace
        onDocumentsChange={handleDocumentsChange}
        onStatsChange={handleStatsChange}
        activeFilename={selectedFilename}
        onAskDocument={handleAskDocument}
      />
      <ChatPanel
        documents={documents}
        stats={stats}
        selectedFilename={selectedFilename}
        onSelectedFilenameChange={setSelectedFilename}
      />
    </div>
  );
}
