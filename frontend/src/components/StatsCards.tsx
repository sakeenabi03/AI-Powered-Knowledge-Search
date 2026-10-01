type StatsCardsProps = {
  totalDocuments: number;
  totalVectors: number;
  collectionName: string;
  isLoading?: boolean;
};

export function StatsCards({
  totalDocuments,
  totalVectors,
  collectionName,
  isLoading = false,
}: StatsCardsProps) {
  const statusLabel = isLoading
    ? "..."
    : totalDocuments > 0
      ? "Ready"
      : "Idle";

  const rows = [
    { label: "Documents", value: isLoading ? "..." : String(totalDocuments) },
    {
      label: "Indexed Passages",
      value: isLoading ? "..." : String(totalVectors),
    },
    {
      label: "Vector Store",
      value: isLoading ? "..." : collectionName || "ChromaDB",
    },
    { label: "Status", value: statusLabel },
  ];

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <h2 className="mb-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
        Knowledge Base
      </h2>
      <div className="grid grid-cols-2 gap-1.5">
        {rows.map((row) => (
          <article
            key={row.label}
            className="rounded-xl border border-slate-100 bg-slate-50 px-2.5 py-2 dark:border-slate-800 dark:bg-slate-950/60"
          >
            <p className="text-[10px] font-medium tracking-wide text-slate-400 uppercase">
              {row.label}
            </p>
            {isLoading ? (
              <div className="mt-1 h-3.5 w-12 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
            ) : (
              <p className="mt-0.5 truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
                {row.value}
              </p>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
