export type DocumentItem = {
  filename: string;
  file_type: string;
  file_size: number;
  indexed_chunks: number;
};

export type DocumentStats = {
  total_documents: number;
  total_vectors: number;
  collection_name: string;
};

export type ChatSource = {
  source_number: number;
  filename: string;
  chunk_index: number;
  chunk_id: string;
  similarity_score: number;
  excerpt: string;
  content: string;
};

export type ChatResponse = {
  query: string;
  answer: string;
  retrieved_count: number;
  sources: ChatSource[];
  model: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  model?: string;
  createdAt: string;
  /** Number of chunks used after similarity filtering (assistant only). */
  retrievedCount?: number;
  isError?: boolean;
};

export type DocumentUploadResponse = {
  filename: string;
  file_type: string;
  file_size: number;
  character_count: number;
  total_chunks: number;
  vector_count: number;
  message: string;
};

export type SearchResult = {
  rank: number;
  chunk_id: string;
  filename: string;
  chunk_index: number;
  content: string;
  character_count: number;
  distance: number;
  similarity_score: number;
};

export type SearchResponse = {
  query: string;
  total_results: number;
  results: SearchResult[];
};
