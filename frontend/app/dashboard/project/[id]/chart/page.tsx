"use client";

import { useState, useEffect, use, useRef, useCallback } from "react";
import { useSession } from "next-auth/react";
import { ChevronRight, ChevronDown, ChevronLeft, File, Folder, Code, Variable, ExternalLink, Loader2, AlertCircle, GitBranch, RefreshCw, ZoomIn, ZoomOut, Maximize2, Move } from "lucide-react";
import Link from "next/link";
import * as d3 from "d3";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { FileNode, FunctionInfo, VariableInfo } from "@/types";
import { getProjectFileTree, getProject, getStructureDiagram, type StructureDiagramNode } from "@/lib/api";

// Helper to convert backend file tree to frontend format
function convertFileTree(nodes: FileNode[]): FileNode[] {
  return nodes.map(node => ({
    ...node,
    type: node.is_dir ? "directory" : "file",
    children: node.children ? convertFileTree(node.children) : undefined,
  }));
}

// D3 Tree Node interface
interface D3TreeNode {
  name: string;
  type: 'directory' | 'file' | 'function' | 'variable';
  path?: string;
  description?: string;
  children?: D3TreeNode[];
  // For tracking original node data
  originalNode?: FileNode;
  originalFunction?: FunctionInfo;
  originalVariable?: VariableInfo;
}

// Convert a single FileNode to D3 tree (with all its contents)
function convertNodeToD3Tree(node: FileNode): D3TreeNode {
  const d3Node: D3TreeNode = {
    name: node.name,
    type: node.is_dir ? "directory" : "file",
    path: node.path,
    originalNode: node,
    children: []
  };
  
  // Add children directories/files
  if (node.children) {
    d3Node.children = node.children.map(convertNodeToD3Tree);
  }
  
  // Add functions as children for files
  if (node.functions && node.functions.length > 0) {
    const functionNodes: D3TreeNode[] = node.functions.map(fn => ({
      name: fn.name + "()",
      type: "function" as const,
      description: fn.description,
      path: node.path,
      originalFunction: fn
    }));
    d3Node.children = [...(d3Node.children || []), ...functionNodes];
  }
  
  // Add variables as children for files
  if (node.variables && node.variables.length > 0) {
    const variableNodes: D3TreeNode[] = node.variables.map(v => ({
      name: v.name,
      type: "variable" as const,
      description: v.description,
      path: node.path,
      originalVariable: v
    }));
    d3Node.children = [...(d3Node.children || []), ...variableNodes];
  }
  
  return d3Node;
}

// Build path from root to a node
function buildPathToNode(fileTree: FileNode[], targetPath: string): D3TreeNode | null {
  function findPath(nodes: FileNode[], path: FileNode[]): FileNode[] | null {
    for (const node of nodes) {
      if (node.path === targetPath) {
        return [...path, node];
      }
      if (node.children) {
        const result = findPath(node.children, [...path, node]);
        if (result) return result;
      }
    }
    return null;
  }
  
  const pathNodes = findPath(fileTree, []);
  if (!pathNodes || pathNodes.length === 0) return null;
  
  // Build the tree from path
  let current: D3TreeNode | null = null;
  
  // Start from target and work backwards
  for (let i = pathNodes.length - 1; i >= 0; i--) {
    const node = pathNodes[i];
    const isTarget = i === pathNodes.length - 1;
    
    if (isTarget) {
      // For the target node, fully expand the entire subtree using convertNodeToD3Tree
      current = convertNodeToD3Tree(node);
    } else {
      // For parent nodes, just create a path node with current as child
      const d3Node: D3TreeNode = {
        name: node.name,
        type: node.is_dir ? "directory" : "file",
        path: node.path,
        originalNode: node,
        children: current ? [current] : []
      };
      current = d3Node;
    }
  }
  
  // Wrap in a root node
  return {
    name: "Project Root",
    type: "directory",
    children: current ? [current] : []
  };
}

// Convert entire file tree to D3 format
function convertFullTreeToD3(nodes: FileNode[]): D3TreeNode {
  return {
    name: "Project",
    type: "directory",
    children: nodes.map(convertNodeToD3Tree)
  };
}

export default function ChartPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const { data: session } = useSession();
  const userId = session?.user?.id;
  
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedTreeNode, setSelectedTreeNode] = useState<FileNode | null>(null);
  const [selectedFunction, setSelectedFunction] = useState<FunctionInfo | null>(null);
  const [selectedVariable, setSelectedVariable] = useState<VariableInfo | null>(null);
  
  // D3 state
  const [nodeMetadata, setNodeMetadata] = useState<Record<string, StructureDiagramNode>>({});
  const [hoveredNode, setHoveredNode] = useState<StructureDiagramNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [diagramLoading, setDiagramLoading] = useState(false);
  
  // Refs
  const d3Ref = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const d3SvgRef = useRef<SVGSVGElement | null>(null);
  const d3ZoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  
  // Zoom state for display
  const [currentZoom, setCurrentZoom] = useState(1);
  
  // Selected D3 node for context panel
  const [selectedD3Node, setSelectedD3Node] = useState<{
    node: D3TreeNode;
    parent: D3TreeNode | null;
    children: D3TreeNode[];
  } | null>(null);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      setError(null);
      
      // Check project status first
      const projectResult = await getProject(resolvedParams.id, userId);
      if (!projectResult.success || !projectResult.data) {
        setError("Project not found");
        setIsLoading(false);
        return;
      }
      
      if (projectResult.data.status !== "ready") {
        setError(`Project is ${projectResult.data.status}. Please analyze it first from the dashboard.`);
        setIsLoading(false);
        return;
      }
      
      // Fetch file tree and structure diagram in parallel
      const [fileTreeResult, diagramResult] = await Promise.all([
        getProjectFileTree(resolvedParams.id, userId),
        getStructureDiagram(resolvedParams.id, userId),
      ]);
      
      if (fileTreeResult.success && fileTreeResult.data) {
        const converted = convertFileTree(fileTreeResult.data.tree);
        setFileTree(converted);
        // Auto-expand first level
        const firstLevelPaths = converted.filter(n => n.type === "directory").map(n => n.path);
        setExpandedNodes(new Set(firstLevelPaths));
      } else {
        setError(fileTreeResult.error || "Failed to load file tree");
      }
      
      if (diagramResult.success && diagramResult.data) {
        setNodeMetadata(diagramResult.data.node_metadata);
      }
      
      setIsLoading(false);
    }
    loadData();
  }, [resolvedParams.id, userId]);
  
  // Render D3 tree diagram
  const renderD3Diagram = useCallback((treeData: D3TreeNode) => {
    if (!d3Ref.current || !containerRef.current) {
      console.warn("D3 container ref not available");
      return;
    }
    
    setDiagramLoading(true);
    
    // Clear previous content
    d3.select(d3Ref.current).selectAll("*").remove();
    
    const containerRect = containerRef.current.getBoundingClientRect();
    const width = containerRect.width || 800;
    const height = containerRect.height || 600;
    
    // Create SVG
    const svg = d3.select(d3Ref.current)
      .append("svg")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .style("cursor", "grab");
    
    d3SvgRef.current = svg.node();
    
    // Create a group for zooming
    const g = svg.append("g").attr("class", "zoom-layer");
    
    // Add a transparent background rect for pan detection
    g.append("rect")
      .attr("class", "pan-background")
      .attr("x", -10000)
      .attr("y", -10000)
      .attr("width", 20000)
      .attr("height", 20000)
      .attr("fill", "transparent");
    
    // Create tree layout - horizontal (root left, children right)
    const treeLayout = d3.tree<D3TreeNode>()
      .nodeSize([60, 280]); // [vertical spacing between siblings, horizontal spacing between levels]
    
    // Create hierarchy
    const root = d3.hierarchy(treeData);
    
    // Apply layout
    const treeRoot = treeLayout(root);
    
    // Calculate bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    treeRoot.each(d => {
      if (d.x < minX) minX = d.x;
      if (d.x > maxX) maxX = d.x;
      if (d.y < minY) minY = d.y;
      if (d.y > maxY) maxY = d.y;
    });
    
    // Store node positions for drag updates
    const nodePositions = new Map<d3.HierarchyPointNode<D3TreeNode>, { x: number; y: number }>();
    treeRoot.each(d => {
      nodePositions.set(d, { x: d.y, y: d.x }); // Swap x/y for horizontal layout
    });
    
    // Create links group
    const linksGroup = g.append("g").attr("class", "links");
    
    // Function to update link paths
    const updateLinks = () => {
      linksGroup.selectAll<SVGPathElement, d3.HierarchyPointLink<D3TreeNode>>(".link")
        .attr("d", d => {
          const sourcePos = nodePositions.get(d.source) || { x: d.source.y, y: d.source.x };
          const targetPos = nodePositions.get(d.target) || { x: d.target.y, y: d.target.x };
          // Smooth bezier curve with control points at 40% and 60% of the horizontal distance
          const midX = (sourcePos.x + targetPos.x) / 2;
          return `M${sourcePos.x + 75},${sourcePos.y}
                  C${midX + 40},${sourcePos.y}
                   ${midX + 40},${targetPos.y}
                   ${targetPos.x - 75},${targetPos.y}`;
        });
    };
    
    // Add links
    linksGroup.selectAll(".link")
      .data(treeRoot.links())
      .enter()
      .append("path")
      .attr("class", "link")
      .attr("fill", "none")
      .attr("stroke", "#4b5563")
      .attr("stroke-width", 1.5);
    
    updateLinks();
    
    // Create drag behavior for nodes - use subject to track starting position
    const dragBehavior = d3.drag<SVGGElement, d3.HierarchyPointNode<D3TreeNode>>()
      .subject(function(event, d) {
        // Return the current position as the drag subject
        const pos = nodePositions.get(d);
        return pos ? { x: pos.x, y: pos.y } : { x: 0, y: 0 };
      })
      .on("start", function(event) {
        event.sourceEvent.stopPropagation(); // Prevent zoom while dragging
        d3.select(this).raise().select("rect").attr("stroke-width", 3);
      })
      .on("drag", function(event, d) {
        const pos = nodePositions.get(d);
        if (pos) {
          // event.x and event.y are now properly relative to the subject
          pos.x = event.x;
          pos.y = event.y;
          d3.select(this).attr("transform", `translate(${event.x},${event.y})`);
          updateLinks();
        }
      })
      .on("end", function() {
        d3.select(this).select("rect").attr("stroke-width", 2);
      });
    
    // Add nodes group
    const nodesGroup = g.append("g").attr("class", "nodes");
    
    // Add nodes
    const nodes = nodesGroup.selectAll(".node")
      .data(treeRoot.descendants())
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", d => {
        const pos = nodePositions.get(d)!;
        return `translate(${pos.x},${pos.y})`;
      })
      .call(dragBehavior as any);
    
    // Add node backgrounds
    nodes.append("rect")
      .attr("x", -70)
      .attr("y", -18)
      .attr("width", 140)
      .attr("height", 36)
      .attr("rx", 6)
      .attr("ry", 6)
      .attr("fill", d => {
        switch (d.data.type) {
          case "directory": return "#3b82f6";
          case "file": return "#6b7280";
          case "function": return "#8b5cf6";
          case "variable": return "#22c55e";
          default: return "#6b7280";
        }
      })
      .attr("stroke", d => {
        switch (d.data.type) {
          case "directory": return "#1d4ed8";
          case "file": return "#4b5563";
          case "function": return "#6d28d9";
          case "variable": return "#16a34a";
          default: return "#4b5563";
        }
      })
      .attr("stroke-width", 2)
      .style("cursor", "grab");
    
    // Add icons
    nodes.append("text")
      .attr("x", -55)
      .attr("y", 5)
      .attr("font-size", "14px")
      .style("pointer-events", "none")
      .text(d => {
        switch (d.data.type) {
          case "directory": return "📁";
          case "file": return "📄";
          case "function": return "⚡";
          case "variable": return "📦";
          default: return "";
        }
      });
    
    // Add labels
    nodes.append("text")
      .attr("x", -40)
      .attr("y", 5)
      .attr("fill", "white")
      .attr("font-size", "11px")
      .attr("font-weight", "500")
      .style("pointer-events", "none")
      .text(d => {
        const name = d.data.name;
        return name.length > 14 ? name.substring(0, 12) + "..." : name;
      })
      .append("title")
      .text(d => d.data.name);
    
    // Add hover and click effects
    nodes
      .on("mouseenter", function(event, d) {
        d3.select(this).select("rect")
          .attr("filter", "drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3))");
        
        const rect = (event.target as Element).getBoundingClientRect();
        setTooltipPosition({ x: rect.left + rect.width / 2, y: rect.top - 10 });
        setHoveredNode({
          id: d.data.name,
          name: d.data.name,
          type: d.data.type,
          description: d.data.description || `${d.data.type}: ${d.data.name}`,
          path: d.data.path,
        });
      })
      .on("mouseleave", function() {
        d3.select(this).select("rect")
          .attr("filter", null);
        setHoveredNode(null);
      })
      .on("click", function(event, d) {
        event.stopPropagation();
        // Set selected node with parent and children info
        setSelectedD3Node({
          node: d.data,
          parent: d.parent?.data || null,
          children: d.data.children || []
        });
        // Highlight selected node
        nodes.selectAll("rect").attr("stroke-width", 2);
        d3.select(this).select("rect").attr("stroke-width", 4);
      });
    
    // Setup D3 zoom - with filter to only pan on background
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .filter((event) => {
        // Allow wheel events for zooming always
        if (event.type === 'wheel') return true;
        // For drag/pan, only allow on background or svg
        const target = event.target as Element;
        const tagName = target.tagName.toLowerCase();
        // Only pan if clicking on SVG itself or the background rect
        return tagName === 'svg' || target.classList.contains('pan-background');
      })
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString());
        setCurrentZoom(event.transform.k);
      })
      .on("start", function(event) {
        const target = event.sourceEvent?.target as Element;
        if (target?.tagName?.toLowerCase() === 'svg' || target?.classList?.contains('pan-background')) {
          svg.style("cursor", "grabbing");
        }
      })
      .on("end", () => {
        svg.style("cursor", "grab");
      });
    
    d3ZoomRef.current = zoomBehavior;
    svg.call(zoomBehavior);
    
    // Initial transform to center and fit
    const treeWidth = maxY - minY + 300;
    const treeHeight = maxX - minX + 100;
    const scale = Math.min(
      (width - 80) / treeWidth,
      (height - 80) / treeHeight,
      0.9
    );
    const tx = 100;
    const ty = (height / 2) - (minX * scale);
    
    svg.call(
      zoomBehavior.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
    
    setDiagramLoading(false);
  }, []);
  
  // Effect to render D3 when selection changes or on initial load
  useEffect(() => {
    if (!fileTree.length) return;
    
    let treeData: D3TreeNode;
    
    if (selectedTreeNode) {
      // Build tree showing path to selected node and its contents
      const pathTree = buildPathToNode(fileTree, selectedTreeNode.path);
      if (pathTree) {
        treeData = pathTree;
      } else {
        treeData = convertFullTreeToD3(fileTree);
      }
    } else {
      // Show full tree
      treeData = convertFullTreeToD3(fileTree);
    }
    
    // Small delay to ensure container is sized
    setTimeout(() => renderD3Diagram(treeData), 50);
  }, [fileTree, selectedTreeNode, renderD3Diagram]);
  
  // Refresh diagram function
  const refreshDiagram = async () => {
    setDiagramLoading(true);
    const [fileTreeResult, diagramResult] = await Promise.all([
      getProjectFileTree(resolvedParams.id, userId),
      getStructureDiagram(resolvedParams.id, userId),
    ]);
    
    if (fileTreeResult.success && fileTreeResult.data) {
      const converted = convertFileTree(fileTreeResult.data.tree);
      setFileTree(converted);
    }
    
    if (diagramResult.success && diagramResult.data) {
      setNodeMetadata(diagramResult.data.node_metadata);
    }
    setDiagramLoading(false);
  };
  
  // Zoom control handlers
  const handleZoomIn = () => {
    if (!d3SvgRef.current || !d3ZoomRef.current) return;
    const svg = d3.select(d3SvgRef.current);
    d3ZoomRef.current.scaleBy(svg.transition().duration(300), 1.5);
  };
  
  const handleZoomOut = () => {
    if (!d3SvgRef.current || !d3ZoomRef.current) return;
    const svg = d3.select(d3SvgRef.current);
    d3ZoomRef.current.scaleBy(svg.transition().duration(300), 0.67);
  };
  
  const handleResetView = () => {
    if (!d3SvgRef.current || !d3ZoomRef.current || !containerRef.current) return;
    const svg = d3.select(d3SvgRef.current);
    const innerGroup = d3SvgRef.current.querySelector("g.zoom-layer");
    if (!innerGroup) return;
    
    const containerRect = containerRef.current.getBoundingClientRect();
    const svgBBox = (innerGroup as SVGGElement).getBBox();
    
    if (svgBBox.width > 0) {
      const scale = Math.min(
        (containerRect.width - 40) / svgBBox.width,
        (containerRect.height - 40) / svgBBox.height,
        1
      );
      const tx = (containerRect.width - svgBBox.width * scale) / 2 - svgBBox.x * scale;
      const ty = (containerRect.height - svgBBox.height * scale) / 2 - svgBBox.y * scale;
      
      svg.transition().duration(500).call(
        d3ZoomRef.current.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
      );
    }
  };

  const toggleNode = (path: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedNodes(newExpanded);
  };

  const handleSelectNode = (node: FileNode) => {
    setSelectedTreeNode(node);
    setSelectedFunction(null);
    setSelectedVariable(null);
    setSelectedD3Node(null);
  };

  const handleSelectFunction = (fn: FunctionInfo, parentNode: FileNode) => {
    setSelectedTreeNode(parentNode);
    setSelectedFunction(fn);
    setSelectedVariable(null);
  };

  const handleSelectVariable = (v: VariableInfo, parentNode: FileNode) => {
    setSelectedTreeNode(parentNode);
    setSelectedVariable(v);
    setSelectedFunction(null);
  };

  const renderTreeNode = (node: FileNode, depth: number = 0) => {
    const isExpanded = expandedNodes.has(node.path);
    const isDirectory = node.type === "directory" || node.is_dir;
    const isSelected = selectedTreeNode?.path === node.path;

    return (
      <div key={node.path}>
        <div
          className={`flex items-center gap-1 py-1.5 px-2 rounded cursor-pointer hover:bg-muted transition-colors ${
            isSelected ? "bg-primary/20 border-l-2 border-primary" : ""
          }`}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
          onClick={() => {
            if (isDirectory) {
              toggleNode(node.path);
            }
            handleSelectNode(node);
          }}
        >
          {isDirectory ? (
            isExpanded ? (
              <ChevronDown className="h-3 w-3 text-muted-foreground flex-shrink-0" />
            ) : (
              <ChevronRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
            )
          ) : (
            <span className="w-3 flex-shrink-0" />
          )}
          {isDirectory ? (
            <Folder className="h-4 w-4 text-blue-500 flex-shrink-0" />
          ) : (
            <File className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          )}
          <span className="text-sm truncate flex-1">{node.name}</span>
          {node.functions && node.functions.length > 0 && (
            <Badge variant="secondary" className="ml-1 text-[10px] px-1 py-0">
              {node.functions.length}
            </Badge>
          )}
        </div>

        {isDirectory && isExpanded && node.children?.map((child) => renderTreeNode(child, depth + 1))}

        {!isDirectory && isExpanded && (
          <div style={{ paddingLeft: `${(depth + 1) * 12 + 8}px` }}>
            {node.functions?.map((fn) => (
              <div
                key={fn.name}
                className={`flex items-center gap-2 py-1 px-2 rounded cursor-pointer hover:bg-muted transition-colors ${
                  selectedFunction?.name === fn.name ? "bg-purple-500/10" : ""
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelectFunction(fn, node);
                }}
              >
                <Code className="h-3 w-3 text-purple-500 flex-shrink-0" />
                <span className="text-xs text-muted-foreground truncate">{fn.name}()</span>
              </div>
            ))}
            {node.variables?.map((v) => (
              <div
                key={v.name}
                className={`flex items-center gap-2 py-1 px-2 rounded cursor-pointer hover:bg-muted transition-colors ${
                  selectedVariable?.name === v.name ? "bg-green-500/10" : ""
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelectVariable(v, node);
                }}
              >
                <Variable className="h-3 w-3 text-green-500 flex-shrink-0" />
                <span className="text-xs text-muted-foreground truncate">{v.name}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <AlertCircle className="h-12 w-12 text-destructive" />
        <p className="text-destructive">{error}</p>
        <Button asChild variant="outline">
          <Link href="/dashboard">Back to Dashboard</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      {/* Left: File Tree */}
      <Card className="w-72 flex-shrink-0 flex flex-col">
        <CardHeader className="pb-2 px-3 pt-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm flex items-center gap-2">
              <Folder className="h-4 w-4" />
              File Structure
            </CardTitle>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-xs"
              onClick={() => {
                setSelectedTreeNode(null);
                setSelectedFunction(null);
                setSelectedVariable(null);
              }}
            >
              Show All
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Select to view in diagram
          </p>
        </CardHeader>
        <CardContent className="p-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <div className="p-2">
              {fileTree.length > 0 ? (
                fileTree.map((node) => renderTreeNode(node))
              ) : (
                <p className="text-sm text-muted-foreground p-4 text-center">No files found</p>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Center: D3 Chart */}
      <Card className="flex-1 min-w-0 flex flex-col">
        <CardHeader className="pb-2 px-4 pt-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm flex items-center gap-2">
                <GitBranch className="h-4 w-4" />
                {selectedTreeNode ? (
                  <span className="flex items-center gap-1">
                    <span className="text-muted-foreground">Path to:</span>
                    <code className="bg-muted px-1 rounded text-xs">{selectedTreeNode.name}</code>
                  </span>
                ) : (
                  "Full Project Structure"
                )}
              </CardTitle>
              <CardDescription className="text-xs">
                Drag nodes to reposition • Scroll to zoom • Click empty area to pan
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {/* Zoom controls */}
              <div className="flex items-center gap-1 border rounded-md p-0.5">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={handleZoomOut}
                  title="Zoom out"
                >
                  <ZoomOut className="h-3 w-3" />
                </Button>
                <span className="text-xs w-10 text-center font-mono">
                  {Math.round(currentZoom * 100)}%
                </span>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={handleZoomIn}
                  title="Zoom in"
                >
                  <ZoomIn className="h-3 w-3" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={handleResetView}
                  title="Reset view"
                >
                  <Maximize2 className="h-3 w-3" />
                </Button>
              </div>
              
              <Button 
                size="sm" 
                variant="outline" 
                className="h-7 text-xs"
                onClick={refreshDiagram}
                disabled={diagramLoading}
              >
                <RefreshCw className={`h-3 w-3 mr-1 ${diagramLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0 flex-1 overflow-hidden">
          <div 
            ref={containerRef}
            className="h-full w-full relative bg-muted/10"
          >
            {/* Loading overlay */}
            {diagramLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            )}
            
            {/* D3 container */}
            <div 
              ref={d3Ref} 
              className="d3-container h-full w-full"
            />
            
            {/* No diagram message */}
            {!fileTree.length && !diagramLoading && (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <GitBranch className="h-12 w-12 mb-4" />
                <p>No diagram available</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Right: Context Panel */}
      <Card className="w-72 flex-shrink-0 flex flex-col">
        <CardHeader className="pb-2 px-3 pt-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            Node Details
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Click a node in the chart
          </p>
        </CardHeader>
        <CardContent className="p-0 flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            {selectedD3Node ? (
              <div className="p-3 space-y-4">
                {/* Parent (Back) Section */}
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1">
                    <ChevronLeft className="h-3 w-3" />
                    Parent (Source)
                  </h4>
                  {selectedD3Node.parent ? (
                    <div className="flex items-center gap-2 p-2 rounded bg-muted/50 border">
                      {selectedD3Node.parent.type === 'directory' && <Folder className="h-4 w-4 text-blue-500" />}
                      {selectedD3Node.parent.type === 'file' && <File className="h-4 w-4 text-gray-500" />}
                      {selectedD3Node.parent.type === 'function' && <Code className="h-4 w-4 text-purple-500" />}
                      {selectedD3Node.parent.type === 'variable' && <Variable className="h-4 w-4 text-green-500" />}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{selectedD3Node.parent.name}</p>
                        <p className="text-xs text-muted-foreground capitalize">{selectedD3Node.parent.type}</p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground italic p-2">Root node</p>
                  )}
                </div>
                
                {/* Selected Node */}
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    Selected
                  </h4>
                  <div className="p-2 rounded bg-primary/10 border-2 border-primary">
                    <div className="flex items-center gap-2">
                      {selectedD3Node.node.type === 'directory' && <Folder className="h-4 w-4 text-blue-500" />}
                      {selectedD3Node.node.type === 'file' && <File className="h-4 w-4 text-gray-500" />}
                      {selectedD3Node.node.type === 'function' && <Code className="h-4 w-4 text-purple-500" />}
                      {selectedD3Node.node.type === 'variable' && <Variable className="h-4 w-4 text-green-500" />}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{selectedD3Node.node.name}</p>
                        <p className="text-xs text-muted-foreground capitalize">{selectedD3Node.node.type}</p>
                      </div>
                    </div>
                    {selectedD3Node.node.path && (
                      <p className="text-xs text-muted-foreground font-mono mt-2 break-all">{selectedD3Node.node.path}</p>
                    )}
                    {selectedD3Node.node.description && (
                      <p className="text-xs text-muted-foreground mt-2">{selectedD3Node.node.description}</p>
                    )}
                  </div>
                </div>
                
                {/* Children (Contents) Section */}
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1">
                    <ChevronRight className="h-3 w-3" />
                    Contents {selectedD3Node.children.length > 0 && `(${selectedD3Node.children.length})`}
                  </h4>
                  {selectedD3Node.children.length > 0 ? (
                    <div className="space-y-1 max-h-64 overflow-y-auto">
                      {selectedD3Node.children.map((child, idx) => (
                        <div key={idx} className="flex items-center gap-2 p-1.5 rounded bg-muted/50 border text-xs">
                          {child.type === 'directory' && <Folder className="h-3 w-3 text-blue-500 flex-shrink-0" />}
                          {child.type === 'file' && <File className="h-3 w-3 text-gray-500 flex-shrink-0" />}
                          {child.type === 'function' && <Code className="h-3 w-3 text-purple-500 flex-shrink-0" />}
                          {child.type === 'variable' && <Variable className="h-3 w-3 text-green-500 flex-shrink-0" />}
                          <span className="truncate">{child.name}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground italic p-2">No children</p>
                  )}
                </div>
              </div>
            ) : selectedFunction ? (
              <div className="p-3 space-y-3">
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    Function
                  </h4>
                  <div className="flex items-center gap-2 p-2 rounded bg-purple-500/10 border border-purple-500/30">
                    <Code className="h-4 w-4 text-purple-500" />
                    <span className="text-sm font-medium">{selectedFunction.name}()</span>
                  </div>
                </div>
                {selectedFunction.description && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      Description
                    </h4>
                    <p className="text-xs text-muted-foreground">{selectedFunction.description}</p>
                  </div>
                )}
                {selectedFunction.parameters && selectedFunction.parameters.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      Parameters
                    </h4>
                    <ul className="space-y-1">
                      {selectedFunction.parameters.map((p) => (
                        <li key={p.name} className="text-xs">
                          <code className="bg-muted px-1 rounded">{p.name}</code>
                          {p.type && <span className="text-muted-foreground">: {p.type}</span>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {selectedFunction.returnType && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      Returns
                    </h4>
                    <code className="bg-muted px-1 rounded text-xs">{selectedFunction.returnType}</code>
                  </div>
                )}
                {selectedTreeNode && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      File
                    </h4>
                    <p className="text-xs text-muted-foreground font-mono break-all">{selectedTreeNode.path}</p>
                  </div>
                )}
              </div>
            ) : selectedVariable ? (
              <div className="p-3 space-y-3">
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    Variable
                  </h4>
                  <div className="flex items-center gap-2 p-2 rounded bg-green-500/10 border border-green-500/30">
                    <Variable className="h-4 w-4 text-green-500" />
                    <span className="text-sm font-medium">{selectedVariable.name}</span>
                  </div>
                </div>
                {selectedVariable.type && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      Type
                    </h4>
                    <code className="bg-muted px-1 rounded text-xs">{selectedVariable.type}</code>
                  </div>
                )}
                {selectedVariable.description && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      Description
                    </h4>
                    <p className="text-xs text-muted-foreground">{selectedVariable.description}</p>
                  </div>
                )}
                {selectedTreeNode && (
                  <div>
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                      File
                    </h4>
                    <p className="text-xs text-muted-foreground font-mono break-all">{selectedTreeNode.path}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-48 text-muted-foreground p-4">
                <GitBranch className="h-8 w-8 mb-2 opacity-50" />
                <p className="text-sm text-center">Select from file tree or click a node in chart</p>
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
      
      {/* Floating Tooltip */}
      {hoveredNode && (
        <div
          className="fixed z-50 bg-popover border rounded-lg shadow-lg p-2 max-w-xs pointer-events-none"
          style={{
            left: tooltipPosition.x,
            top: tooltipPosition.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div className="flex items-center gap-2 mb-1">
            {hoveredNode.type === "function" && <Code className="h-3 w-3 text-purple-500" />}
            {hoveredNode.type === "variable" && <Variable className="h-3 w-3 text-green-500" />}
            {hoveredNode.type === "file" && <File className="h-3 w-3 text-gray-500" />}
            {hoveredNode.type === "directory" && <Folder className="h-3 w-3 text-blue-500" />}
            <span className="font-semibold text-xs">{hoveredNode.name}</span>
            <Badge variant="secondary" className="text-[10px] capitalize">{hoveredNode.type}</Badge>
          </div>
          {hoveredNode.description && (
            <p className="text-xs text-muted-foreground">{hoveredNode.description}</p>
          )}
        </div>
      )}
    </div>
  );
}
