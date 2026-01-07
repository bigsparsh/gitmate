import { redirect } from "next/navigation";

export default function ProjectPage({ params }: { params: { id: string } }) {
  // Redirect to chart view by default
  redirect(`/dashboard/project/${params.id}/chart`);
}
