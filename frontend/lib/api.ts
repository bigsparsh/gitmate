import axios, { type AxiosRequestConfig } from "axios";
import type {
  Project,
  ProjectAnalysis,
  ChatMessage,
  ApiResponse,
  FunctionInfo,
  ReferenceInfo,
  CallInfo,
  FileNode,
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Helper to add user ID header
function withUserId(userId?: string): AxiosRequestConfig {
  if (!userId) return {};
  return {
    headers: {
      "X-User-Id": userId,
    },
  };
}

// Project APIs
export async function createProject(
  repoUrl: string,
  name: string,
  userId?: string,
  description?: string
): Promise<ApiResponse<Project>> {
  try {
    const response = await api.post(
      "/api/projects",
      { repo_url: repoUrl, name, description },
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getProjects(userId?: string): Promise<ApiResponse<Project[]>> {
  try {
    const response = await api.get("/api/projects", withUserId(userId));
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getProject(id: string, userId?: string): Promise<ApiResponse<Project>> {
  try {
    const response = await api.get(`/api/projects/${id}`, withUserId(userId));
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function deleteProject(id: string, userId?: string): Promise<ApiResponse<void>> {
  try {
    await api.delete(`/api/projects/${id}`, withUserId(userId));
    return { success: true };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Analysis APIs
export async function analyzeProject(
  projectId: string,
  userId?: string,
  options?: { skipLsp?: boolean; skipLlmAnalysis?: boolean }
): Promise<ApiResponse<{ success: boolean; message: string; project: Project }>> {
  try {
    const params = new URLSearchParams();
    if (options?.skipLsp) params.append("skip_lsp", "false");
    if (options?.skipLlmAnalysis) params.append("skip_llm_analysis", "false");
    
    const response = await api.post(
      `/api/projects/${projectId}/analyze?${params.toString()}`,
      {},
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getProjectStats(
  projectId: string,
  userId?: string
): Promise<ApiResponse<ProjectAnalysis>> {
  try {
    const response = await api.get(`/api/projects/${projectId}/stats`, withUserId(userId));
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getProjectFileTree(
  projectId: string,
  userId?: string
): Promise<ApiResponse<{ repo_path: string; tree: FileNode[] }>> {
  try {
    const response = await api.get(`/api/projects/${projectId}/file-tree`, withUserId(userId));
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getProjectEntities(
  projectId: string,
  userId?: string,
  options?: { entityType?: string; filePath?: string; search?: string; limit?: number; offset?: number }
): Promise<ApiResponse<{ total: number; entities: FunctionInfo[] }>> {
  try {
    const params = new URLSearchParams();
    if (options?.entityType) params.append("entity_type", options.entityType);
    if (options?.filePath) params.append("file_path", options.filePath);
    if (options?.search) params.append("search", options.search);
    if (options?.limit) params.append("limit", options.limit.toString());
    if (options?.offset) params.append("offset", options.offset.toString());
    
    const response = await api.get(
      `/api/projects/${projectId}/entities?${params.toString()}`,
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Function References and Calls (LSP-based)
export async function getFunctionReferences(
  projectId: string,
  functionName: string,
  userId?: string
): Promise<ApiResponse<ReferenceInfo[]>> {
  try {
    const response = await api.get(
      `/api/projects/${projectId}/refs/${encodeURIComponent(functionName)}`,
      withUserId(userId)
    );
    return { success: true, data: response.data.results };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getFunctionCalls(
  projectId: string,
  functionName: string,
  userId?: string
): Promise<ApiResponse<CallInfo[]>> {
  try {
    const response = await api.get(
      `/api/projects/${projectId}/calls/${encodeURIComponent(functionName)}`,
      withUserId(userId)
    );
    return { success: true, data: response.data.results };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Chart APIs
export async function getCallGraph(
  projectId: string,
  userId?: string,
  focus?: string
): Promise<ApiResponse<{ chart_type: string; chart_code: string }>> {
  try {
    const params = focus ? `?focus=${encodeURIComponent(focus)}` : "";
    const response = await api.get(
      `/api/projects/${projectId}/charts/call-graph${params}`,
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getEntityDiagram(
  projectId: string,
  userId?: string
): Promise<ApiResponse<{ chart_type: string; chart_code: string }>> {
  try {
    const response = await api.get(
      `/api/projects/${projectId}/charts/entity-diagram`,
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Structure Diagram with node metadata for tooltips
export interface StructureDiagramNode {
  id: string;
  name: string;
  type: "root" | "directory" | "file" | "function" | "variable";
  path?: string;
  description: string;
  startLine?: number;
  endLine?: number;
  functionCount?: number;
  variableCount?: number;
  references?: number;
  calls?: number;
  calledBy?: number;
}

export interface StructureDiagramData {
  chart_type: string;
  chart_code: string;
  node_metadata: Record<string, StructureDiagramNode>;
}

export async function getStructureDiagram(
  projectId: string,
  userId?: string
): Promise<ApiResponse<StructureDiagramData>> {
  try {
    const response = await api.get(
      `/api/projects/${projectId}/charts/structure-diagram`,
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// File Content
export async function getFileContent(
  projectId: string,
  filePath: string,
  userId?: string
): Promise<ApiResponse<{ file_path: string; content: string; entities: FunctionInfo[] }>> {
  try {
    const response = await api.get(
      `/api/projects/${projectId}/file/${encodeURIComponent(filePath)}`,
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Chat APIs
export interface CommandResponse {
  type: "refs" | "calls" | "search";
  entity_name?: string;
  function_name?: string;
  query?: string;
  error?: string;
  results: unknown[];
}

export async function sendChatMessage(
  projectId: string,
  message: string,
  userId?: string,
  onChunk?: (chunk: string) => void
): Promise<ApiResponse<ChatMessage> & { commandResponse?: CommandResponse }> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (userId) headers["X-User-Id"] = userId;
    
    const response = await fetch(
      `${API_BASE_URL}/api/projects/${projectId}/chat`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({ message }),
      }
    );

    if (!response.ok) {
      throw new Error("Failed to send message");
    }

    // Check if this is a command response (JSON)
    const contentType = response.headers.get("Content-Type");
    const isCommand = response.headers.get("X-Response-Type") === "command" || 
                      contentType?.includes("application/json");
    
    if (isCommand) {
      const commandData = await response.json() as CommandResponse;
      return {
        success: true,
        commandResponse: commandData,
        data: {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "", // Command responses don't have regular content
          timestamp: new Date(),
        },
      };
    }

    // Regular streaming response
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let fullContent = "";

    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullContent += chunk;

        if (onChunk) {
          onChunk(chunk);
        }
      }
    }

    return {
      success: true,
      data: {
        id: crypto.randomUUID(),
        role: "assistant",
        content: fullContent,
        timestamp: new Date(),
      },
    };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function getChatHistory(
  projectId: string,
  userId?: string
): Promise<ApiResponse<ChatMessage[]>> {
  try {
    const response = await api.get(
      `/api/projects/${projectId}/chat/history`,
      withUserId(userId)
    );
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

export async function clearChatHistory(
  projectId: string,
  userId?: string
): Promise<ApiResponse<void>> {
  try {
    await api.post(`/api/projects/${projectId}/chat/clear`, {}, withUserId(userId));
    return { success: true };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Health check
export async function checkHealth(): Promise<ApiResponse<{ status: string }>> {
  try {
    const response = await api.get("/api/health");
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: getErrorMessage(error) };
  }
}

// Helper function
function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.response?.data?.message || error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "An unexpected error occurred";
}

export default api;
