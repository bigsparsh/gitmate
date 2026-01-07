// Project Types
export type ProjectStatus = "pending" | "analyzing" | "ready" | "error";

export interface Project {
  id: string;
  name: string;
  repoUrl: string;
  description?: string;
  status: ProjectStatus;
  createdAt: Date | string;
  updatedAt: Date | string;
  userId: string;
  totalEntities?: number;
  stats?: Record<string, unknown>;
}

// File Tree Types
export interface FileNode {
  name: string;
  path: string;
  type?: "file" | "directory";
  is_dir?: boolean;
  children?: FileNode[];
  functions?: FunctionInfo[];
  variables?: VariableInfo[];
}

export interface FunctionInfo {
  name: string;
  description?: string;
  startLine: number;
  endLine: number;
  parameters?: ParameterInfo[];
  returnType?: string;
  references?: ReferenceInfo[];
  calls?: CallInfo[];
}

export interface VariableInfo {
  name: string;
  description?: string;
  type?: string;
  line: number;
}

export interface ParameterInfo {
  name: string;
  type?: string;
}

export interface ReferenceInfo {
  filePath: string;
  line: number;
  column: number;
  context?: string;
}

export interface CallInfo {
  name: string;
  filePath: string;
  line: number;
}

// Chat Types
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface ProjectAnalysis {
  projectId: string;
  fileTree: FileNode[];
  totalFiles: number;
  totalFunctions: number;
  analyzedAt: Date;
}

// Chart Node Types for Mermaid/Interactive Charts
export interface ChartNode {
  id: string;
  label: string;
  type: "file" | "function" | "variable";
  description?: string;
  children?: ChartNode[];
  parentId?: string;
}
