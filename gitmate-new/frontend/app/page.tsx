import { signIn } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Github, Chrome, GitBranch, MessageSquare, FileCode, Sparkles } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      {/* Header */}
      <header className="container mx-auto px-4 py-6">
        <nav className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="h-8 w-8 text-primary" />
            <span className="text-2xl font-bold">GitMate</span>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-8">
            <Sparkles className="h-4 w-4" />
            AI-Powered Codebase Onboarding
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6">
            Understand Any GitHub
            <br />
            <span className="text-primary">Codebase Instantly</span>
          </h1>
          
          <p className="text-xl text-muted-foreground mb-12 max-w-2xl mx-auto">
            GitMate analyzes your repository, maps functions and variables, and lets you 
            chat with an AI that understands your code. Perfect for onboarding new developers.
          </p>

          {/* Auth Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <form
              action={async () => {
                "use server";
                await signIn("github", { redirectTo: "/dashboard" });
              }}
            >
              <Button size="lg" className="w-full sm:w-auto gap-2">
                <Github className="h-5 w-5" />
                Continue with GitHub
              </Button>
            </form>
            <form
              action={async () => {
                "use server";
                await signIn("google", { redirectTo: "/dashboard" });
              }}
            >
              <Button size="lg" variant="outline" className="w-full sm:w-auto gap-2">
                <Chrome className="h-5 w-5" />
                Continue with Google
              </Button>
            </form>
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-32 grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <FeatureCard
            icon={<FileCode className="h-10 w-10" />}
            title="Visual Code Structure"
            description="Interactive charts showing your file hierarchy, functions, and their relationships at a glance."
          />
          <FeatureCard
            icon={<MessageSquare className="h-10 w-10" />}
            title="AI Chat Assistant"
            description="Ask questions about your codebase and get intelligent answers with precise code references."
          />
          <FeatureCard
            icon={<GitBranch className="h-10 w-10" />}
            title="LSP-Powered Analysis"
            description="Deep code understanding using Language Server Protocol for accurate function references and call hierarchy."
          />
        </div>

        {/* How it works */}
        <div className="mt-32 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          <div className="grid md:grid-cols-4 gap-6">
            <StepCard number={1} title="Add Repository" description="Paste your GitHub repo URL" />
            <StepCard number={2} title="Analysis" description="AI analyzes all functions & variables" />
            <StepCard number={3} title="Explore" description="View interactive code charts" />
            <StepCard number={4} title="Chat" description="Ask anything about the code" />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 mt-20 border-t">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <GitBranch className="h-5 w-5" />
            <span>GitMate</span>
          </div>
          <p>© 2024 GitMate. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="p-6 rounded-xl border bg-card hover:shadow-lg transition-shadow">
      <div className="text-primary mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}

function StepCard({
  number,
  title,
  description,
}: {
  number: number;
  title: string;
  description: string;
}) {
  return (
    <div className="text-center">
      <div className="w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-xl font-bold mx-auto mb-4">
        {number}
      </div>
      <h3 className="font-semibold mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
