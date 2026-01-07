"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { Plus, FolderGit2, ExternalLink, Trash2, BarChart3, MessageSquare, Loader2, RefreshCw, AlertCircle } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type { Project, ProjectStatus } from "@/types";
import { getProjects, createProject, deleteProject, analyzeProject } from "@/lib/api";

function getStatusBadge(status: ProjectStatus) {
  switch (status) {
    case "pending":
      return <Badge variant="secondary">Pending</Badge>;
    case "analyzing":
      return <Badge variant="outline" className="animate-pulse">Analyzing...</Badge>;
    case "ready":
      return <Badge variant="default" className="bg-green-600">Ready</Badge>;
    case "error":
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
}

export default function DashboardPage() {
  const { data: session } = useSession();
  const [projects, setProjects] = useState<Project[]>([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newProjectUrl, setNewProjectUrl] = useState("");
  const [newProjectName, setNewProjectName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [analyzingProjects, setAnalyzingProjects] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const userId = session?.user?.id;

  // Load projects on mount
  useEffect(() => {
    async function loadProjects() {
      setIsLoadingProjects(true);
      setError(null);
      const result = await getProjects(userId);
      if (result.success && result.data) {
        setProjects(result.data);
      } else {
        setError(result.error || "Failed to load projects");
      }
      setIsLoadingProjects(false);
    }
    loadProjects();
  }, [userId]);

  const handleCreateProject = async () => {
    if (!newProjectUrl || !newProjectName) return;
    
    setIsLoading(true);
    setError(null);
    
    const result = await createProject(newProjectUrl, newProjectName, userId);
    
    if (result.success && result.data) {
      setProjects([result.data, ...projects]);
      setNewProjectUrl("");
      setNewProjectName("");
      setIsDialogOpen(false);
    } else {
      setError(result.error || "Failed to create project");
    }
    
    setIsLoading(false);
  };

  const handleAnalyzeProject = async (projectId: string) => {
    setAnalyzingProjects(prev => new Set(prev).add(projectId));
    
    // Update local state to show analyzing
    setProjects(prev => prev.map(p => 
      p.id === projectId ? { ...p, status: "analyzing" as ProjectStatus } : p
    ));
    
    const result = await analyzeProject(projectId, userId, { skipLsp: true });
    
    if (result.success && result.data?.project) {
      setProjects(prev => prev.map(p => 
        p.id === projectId ? result.data!.project : p
      ));
    } else {
      // Update status to error
      setProjects(prev => prev.map(p => 
        p.id === projectId ? { ...p, status: "error" as ProjectStatus } : p
      ));
      setError(result.error || "Failed to analyze project");
    }
    
    setAnalyzingProjects(prev => {
      const next = new Set(prev);
      next.delete(projectId);
      return next;
    });
  };

  const handleDeleteProject = async (id: string) => {
    const result = await deleteProject(id, userId);
    if (result.success) {
      setProjects(projects.filter((p) => p.id !== id));
    } else {
      setError(result.error || "Failed to delete project");
    }
  };

  if (isLoadingProjects) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Projects</h1>
          <p className="text-muted-foreground">
            Manage your GitHub repositories and explore their codebases
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              New Project
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Project</DialogTitle>
              <DialogDescription>
                Enter a GitHub repository URL to analyze and explore its codebase.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label htmlFor="name" className="text-sm font-medium">
                  Project Name
                </label>
                <Input
                  id="name"
                  placeholder="my-project"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="url" className="text-sm font-medium">
                  Repository URL
                </label>
                <Input
                  id="url"
                  placeholder="https://github.com/username/repo"
                  value={newProjectUrl}
                  onChange={(e) => setNewProjectUrl(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateProject} disabled={isLoading}>
                {isLoading ? "Creating..." : "Create Project"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {projects.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <FolderGit2 className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No projects yet</h3>
            <p className="text-muted-foreground text-center mb-4">
              Add your first GitHub repository to get started
            </p>
            <Button onClick={() => setIsDialogOpen(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              Add Project
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card key={project.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <FolderGit2 className="h-5 w-5 text-primary" />
                    <CardTitle className="text-lg">{project.name}</CardTitle>
                  </div>
                  {getStatusBadge(project.status)}
                </div>
                <CardDescription className="line-clamp-2">
                  {project.description || project.repoUrl}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <ExternalLink className="h-4 w-4" />
                  <a
                    href={project.repoUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline truncate"
                  >
                    {project.repoUrl.replace("https://github.com/", "")}
                  </a>
                </div>
                {project.status === "ready" && project.totalEntities && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {project.totalEntities} entities analyzed
                  </p>
                )}
                {project.status === "error" && (
                  <div className="flex items-center gap-1 text-xs text-destructive mt-2">
                    <AlertCircle className="h-3 w-3" />
                    Analysis failed
                  </div>
                )}
              </CardContent>
              <CardFooter className="flex justify-between">
                <div className="flex gap-2">
                  {project.status === "pending" || project.status === "error" ? (
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="gap-1"
                      onClick={() => handleAnalyzeProject(project.id)}
                      disabled={analyzingProjects.has(project.id)}
                    >
                      {analyzingProjects.has(project.id) ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4" />
                      )}
                      {project.status === "error" ? "Retry" : "Analyze"}
                    </Button>
                  ) : project.status === "analyzing" ? (
                    <Button size="sm" variant="outline" className="gap-1" disabled>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </Button>
                  ) : (
                    <>
                      <Button asChild size="sm" variant="outline" className="gap-1">
                        <Link href={`/dashboard/project/${project.id}/chart`}>
                          <BarChart3 className="h-4 w-4" />
                          Chart
                        </Link>
                      </Button>
                      <Button asChild size="sm" variant="outline" className="gap-1">
                        <Link href={`/dashboard/project/${project.id}/chat`}>
                          <MessageSquare className="h-4 w-4" />
                          Chat
                        </Link>
                      </Button>
                    </>
                  )}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-destructive hover:text-destructive"
                  onClick={() => handleDeleteProject(project.id)}
                  disabled={analyzingProjects.has(project.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      {error && (
        <div className="fixed bottom-4 right-4 bg-destructive text-destructive-foreground px-4 py-2 rounded-md shadow-lg flex items-center gap-2">
          <AlertCircle className="h-4 w-4" />
          {error}
          <button onClick={() => setError(null)} className="ml-2 hover:opacity-80">×</button>
        </div>
      )}
    </div>
  );
}
