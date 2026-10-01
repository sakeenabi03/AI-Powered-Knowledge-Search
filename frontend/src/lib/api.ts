import type {
  ChatResponse,
  DocumentItem,
  DocumentStats,
  DocumentUploadResponse,
  SearchResponse,
} from "@/types";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const ASK_TIMEOUT_MS = 60_000;

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  DEFAULT_API_BASE_URL;

type DocumentDeleteResponse = {
  message: string;
  filename: string;
  deleted_vectors: number;
};

function extractErrorDetail(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return null;
      })
      .filter((item): item is string => Boolean(item));
    if (messages.length > 0) {
      return messages.join(" ");
    }
  }

  return fallback;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  options?: { timeoutMs?: number },
): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = options?.timeoutMs;
  const timeoutId =
    typeof timeoutMs === "number"
      ? globalThis.setTimeout(() => controller.abort(), timeoutMs)
      : null;

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out after 60 seconds.");
    }
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Request timed out after 60 seconds.");
    }
    throw new Error("Unable to connect to the backend service.");
  } finally {
    if (timeoutId !== null) {
      globalThis.clearTimeout(timeoutId);
    }
  }

  if (!response.ok) {
    const fallback = `Request failed (${response.status}).`;
    let detail = fallback;
    try {
      const payload: unknown = await response.json();
      detail = extractErrorDetail(payload, fallback);
    } catch {
      // Keep the default error message when the body is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export async function getDocuments(): Promise<DocumentItem[]> {
  return requestJson<DocumentItem[]>("/api/documents/list");
}

export async function getStats(): Promise<DocumentStats> {
  return requestJson<DocumentStats>("/api/documents/stats");
}

export async function uploadDocument(
  file: File,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<DocumentUploadResponse>("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function deleteDocument(
  filename: string,
): Promise<DocumentDeleteResponse> {
  return requestJson<DocumentDeleteResponse>(
    `/api/documents/${encodeURIComponent(filename)}`,
    {
      method: "DELETE",
    },
  );
}

export async function askQuestion(
  query: string,
  topK: number = 5,
  filename?: string | null,
): Promise<ChatResponse> {
  return requestJson<ChatResponse>(
    "/api/chat/ask",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        top_k: topK,
        filename: filename ?? null,
      }),
    },
    { timeoutMs: ASK_TIMEOUT_MS },
  );
}

export async function searchDocuments(
  query: string,
  topK?: number,
  filename?: string,
): Promise<SearchResponse> {
  return requestJson<SearchResponse>("/api/chat/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      top_k: topK ?? 5,
      filename: filename || undefined,
    }),
  });
}
