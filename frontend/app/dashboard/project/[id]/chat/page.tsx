"use client";

import { useState, useRef, useEffect, use } from "react";
import { useSession } from "next-auth/react";
import { Send, Bot, User, Loader2, BarChart3, AlertCircle, FileCode, ArrowDownLeft, ArrowUpRight, Search, MapPin } from "lucide-react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import type { ChatMessage, CommandResponseData, RefResult, CallHierarchyResult, SearchResultItem } from "@/types";
import { sendChatMessage, getProject, getChatHistory, type CommandResponse } from "@/lib/api";

export default function ChatPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const { data: session } = useSession();
  const userId = session?.user?.id;
  
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      role: "assistant",
      content: `Hello! I'm your AI assistant for this codebase. I can help you understand the code structure, find specific functions, and answer questions about how different parts work together.

Try asking me things like:
- "Where is the authentication logic?"
- "How does the main function work?"
- "What functions are in this file?"

You can also use special commands:
- \`/refs functionName\` - Find all references to a function or entity
- \`/calls functionName\` - Show the call hierarchy for a function
- \`/search query\` - Search for relevant code entities`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load project status and chat history on mount
  useEffect(() => {
    async function initChat() {
      setIsInitializing(true);
      setError(null);
      
      // Check project status
      const projectResult = await getProject(resolvedParams.id, userId);
      if (!projectResult.success || !projectResult.data) {
        setError("Project not found");
        setIsInitializing(false);
        return;
      }
      
      if (projectResult.data.status !== "ready") {
        setError(`Project is ${projectResult.data.status}. Please analyze it first from the dashboard.`);
        setIsInitializing(false);
        return;
      }
      
      // Load existing chat history
      const historyResult = await getChatHistory(resolvedParams.id, userId);
      if (historyResult.success && historyResult.data && historyResult.data.length > 0) {
        const historyMessages: ChatMessage[] = historyResult.data.map((msg, idx) => ({
          id: `history-${idx}`,
          role: msg.role as "user" | "assistant",
          content: msg.content,
          timestamp: new Date(),
        }));
        setMessages(prev => [...prev, ...historyMessages]);
      }
      
      setIsInitializing(false);
    }
    initChat();
  }, [resolvedParams.id, userId]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages, streamingContent]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setStreamingContent("");

    // Use real API
    const result = await sendChatMessage(
      resolvedParams.id,
      userMessage.content,
      userId,
      (chunk) => {
        setStreamingContent(prev => prev + chunk);
      }
    );

    if (result.success) {
      // Check if this is a command response
      if (result.commandResponse) {
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "", // Command responses use special rendering
          timestamp: new Date(),
          commandResponse: result.commandResponse as CommandResponseData,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else if (result.data) {
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: result.data.content,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } else {
      // Show error message
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${result.error || "Unknown error"}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
    
    setStreamingContent("");
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (isInitializing) {
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
    <div className="flex flex-col h-[calc(100vh-10rem)]">
      <Card className="flex-1 flex flex-col">
        <CardHeader className="pb-3 border-b">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              <Bot className="h-5 w-5" />
              AI Chat Assistant
            </CardTitle>
            <Button asChild size="sm" variant="outline">
              <Link href={`/dashboard/project/${resolvedParams.id}/chart`}>
                <BarChart3 className="h-4 w-4 mr-2" />
                View Chart
              </Link>
            </Button>
          </div>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-0">
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            <div className="space-y-4">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              {streamingContent && (
                <MessageBubble
                  message={{
                    id: "streaming",
                    role: "assistant",
                    content: streamingContent,
                    timestamp: new Date(),
                  }}
                  isStreaming
                />
              )}
              {isLoading && !streamingContent && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              )}
            </div>
          </ScrollArea>

          <form onSubmit={handleSubmit} className="p-4 border-t">
            <div className="flex gap-2">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the codebase... (Enter to send, Shift+Enter for new line)"
                className="min-h-[60px] max-h-[200px] resize-none"
                disabled={isLoading}
              />
              <Button type="submit" size="icon" disabled={isLoading || !input.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

function MessageBubble({
  message,
  isStreaming = false,
}: {
  message: ChatMessage;
  isStreaming?: boolean;
}) {
  const isUser = message.role === "user";

  // Check if this is a command response
  if (message.commandResponse) {
    return (
      <div className="flex gap-3">
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarFallback className="bg-muted">
            <Bot className="h-4 w-4" />
          </AvatarFallback>
        </Avatar>
        <div className="flex-1 max-w-[90%]">
          <CommandResponseCard response={message.commandResponse} />
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar className="h-8 w-8 flex-shrink-0">
        <AvatarFallback className={isUser ? "bg-primary text-primary-foreground" : "bg-muted"}>
          {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>
      <div
        className={`rounded-lg px-4 py-2 max-w-[80%] ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || "");
                  const isInline = !match;
                  
                  return isInline ? (
                    <code className="bg-background/50 px-1 py-0.5 rounded text-sm" {...props}>
                      {children}
                    </code>
                  ) : (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
        )}
      </div>
    </div>
  );
}

function CommandResponseCard({ response }: { response: CommandResponseData }) {
  if (response.type === "refs") {
    return <RefsCard entityName={response.entity_name} error={response.error} results={response.results} />;
  }
  if (response.type === "calls") {
    return <CallsCard functionName={response.function_name} error={response.error} results={response.results} />;
  }
  if (response.type === "search") {
    return <SearchCard query={response.query} results={response.results} />;
  }
  return null;
}

function RefsCard({ entityName, error, results }: { entityName: string; error?: string; results: RefResult[] }) {
  if (error) {
    return (
      <Card className="border-yellow-500/50">
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2 text-yellow-600">
            <AlertCircle className="h-4 w-4" />
            No References Found
          </CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          <p className="text-sm text-muted-foreground">{error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <MapPin className="h-4 w-4 text-blue-500" />
        References for &quot;{entityName}&quot;
      </div>
      {results.map((result, idx) => (
        <Card key={idx} className="border-blue-500/30">
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <FileCode className="h-4 w-4 text-blue-500" />
                <span className="font-mono">{result.name}</span>
              </CardTitle>
              <Badge variant="secondary" className="text-xs">{result.entity_type}</Badge>
            </div>
          </CardHeader>
          <CardContent className="py-2 space-y-2">
            <p className="text-xs text-muted-foreground">
              {result.file_path}:{result.line}
            </p>
            
            {/* For functions: show callers and callees */}
            {result.callers && result.callers.length > 0 && (
              <div>
                <p className="text-xs font-medium text-green-600 mb-1">
                  <ArrowDownLeft className="h-3 w-3 inline mr-1" />
                  Called by ({result.callers.length}):
                </p>
                <div className="space-y-1 pl-4">
                  {result.callers.slice(0, 5).map((caller, i) => (
                    <p key={i} className="text-xs text-muted-foreground font-mono">
                      ← {caller.name} ({caller.file_path}:{caller.line})
                    </p>
                  ))}
                  {result.callers.length > 5 && (
                    <p className="text-xs text-muted-foreground">... and {result.callers.length - 5} more</p>
                  )}
                </div>
              </div>
            )}
            
            {result.callees && result.callees.length > 0 && (
              <div>
                <p className="text-xs font-medium text-yellow-600 mb-1">
                  <ArrowUpRight className="h-3 w-3 inline mr-1" />
                  Calls ({result.callees.length}):
                </p>
                <div className="space-y-1 pl-4">
                  {result.callees.slice(0, 5).map((callee, i) => (
                    <p key={i} className="text-xs text-muted-foreground font-mono">
                      → {callee.name} ({callee.file_path}:{callee.line})
                    </p>
                  ))}
                  {result.callees.length > 5 && (
                    <p className="text-xs text-muted-foreground">... and {result.callees.length - 5} more</p>
                  )}
                </div>
              </div>
            )}
            
            {/* For non-functions: show references */}
            {result.references && result.references.length > 0 && (
              <div>
                <p className="text-xs font-medium text-blue-600 mb-1">
                  References ({result.references.length}):
                </p>
                <div className="space-y-1 pl-4">
                  {result.references.slice(0, 10).map((ref, i) => (
                    <p key={i} className="text-xs text-muted-foreground font-mono">
                      • {ref.file_path}:{ref.line}
                    </p>
                  ))}
                  {result.references.length > 10 && (
                    <p className="text-xs text-muted-foreground">... and {result.references.length - 10} more</p>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function CallsCard({ functionName, error, results }: { functionName: string; error?: string; results: CallHierarchyResult[] }) {
  if (error) {
    return (
      <Card className="border-yellow-500/50">
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2 text-yellow-600">
            <AlertCircle className="h-4 w-4" />
            No Call Hierarchy Found
          </CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          <p className="text-sm text-muted-foreground">{error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <BarChart3 className="h-4 w-4 text-purple-500" />
        Call Hierarchy for &quot;{functionName}&quot;
      </div>
      {results.map((result, idx) => (
        <Card key={idx} className="border-purple-500/30">
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <FileCode className="h-4 w-4 text-purple-500" />
                <span className="font-mono">{result.name}</span>
              </CardTitle>
              <Badge variant="secondary" className="text-xs">{result.entity_type}</Badge>
            </div>
          </CardHeader>
          <CardContent className="py-2 space-y-3">
            <p className="text-xs text-muted-foreground">
              {result.file_path}:{result.line}
            </p>
            
            {/* Incoming calls (who calls this function) */}
            <div>
              <p className="text-xs font-medium text-green-600 mb-1">
                <ArrowDownLeft className="h-3 w-3 inline mr-1" />
                Called by ({result.incoming_calls.length}):
              </p>
              {result.incoming_calls.length > 0 ? (
                <div className="space-y-1 pl-4">
                  {result.incoming_calls.map((caller, i) => (
                    <p key={i} className="text-xs text-muted-foreground font-mono">
                      ← <span className="text-cyan-500">{caller.name}</span> ({caller.file_path}:{caller.line})
                    </p>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground pl-4">No callers found</p>
              )}
            </div>
            
            {/* Outgoing calls (what this function calls) */}
            <div>
              <p className="text-xs font-medium text-yellow-600 mb-1">
                <ArrowUpRight className="h-3 w-3 inline mr-1" />
                Calls ({result.outgoing_calls.length}):
              </p>
              {result.outgoing_calls.length > 0 ? (
                <div className="space-y-1 pl-4">
                  {result.outgoing_calls.map((callee, i) => (
                    <p key={i} className="text-xs text-muted-foreground font-mono">
                      → <span className="text-cyan-500">{callee.name}</span> ({callee.file_path}:{callee.line})
                    </p>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground pl-4">No outgoing calls found</p>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function SearchCard({ query, results }: { query: string; results: SearchResultItem[] }) {
  if (results.length === 0) {
    return (
      <Card className="border-yellow-500/50">
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2 text-yellow-600">
            <Search className="h-4 w-4" />
            No Results Found
          </CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          <p className="text-sm text-muted-foreground">No code entities matching &quot;{query}&quot;</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Search className="h-4 w-4 text-green-500" />
        Search Results for &quot;{query}&quot;
      </div>
      {results.map((result, idx) => (
        <Card key={idx} className="border-green-500/30">
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <FileCode className="h-4 w-4 text-green-500" />
                <span className="font-mono">{result.name}</span>
              </CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">{result.entity_type}</Badge>
                <Badge variant="outline" className="text-xs">{(result.relevance_score * 100).toFixed(0)}%</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="py-2">
            <p className="text-xs text-muted-foreground">
              {result.file_path} (lines {result.start_line}-{result.end_line})
            </p>
            <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
              {result.num_references > 0 && <span>References: {result.num_references}</span>}
              {result.num_callers > 0 && <span>Callers: {result.num_callers}</span>}
              {result.num_callees > 0 && <span>Calls: {result.num_callees}</span>}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
