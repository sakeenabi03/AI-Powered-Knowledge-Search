export function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    const kb = bytes / 1024;
    return `${kb >= 100 ? Math.round(kb) : kb.toFixed(kb >= 10 ? 0 : 1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatModelName(model?: string | null): string | null {
  if (!model) {
    return null;
  }

  const raw = model.includes("/") ? (model.split("/").pop() ?? model) : model;
  return raw
    .split("-")
    .map((part) => {
      const lower = part.toLowerCase();
      if (lower === "gpt") {
        return "GPT";
      }
      if (lower === "mini") {
        return "Mini";
      }
      if (/^\d/.test(part)) {
        return part.toUpperCase();
      }
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}
