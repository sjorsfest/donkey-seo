"""Cluster agent for Step 6: Keyword Clustering."""

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent


class ClusterValidation(BaseModel):
    """Validation result for a single cluster."""

    cluster_index: int
    name: str = Field(description="Descriptive name (2-5 words)")
    description: str = Field(description="Brief description of what this cluster covers")
    primary_keyword: str = Field(description="Best primary keyword for this cluster")
    coherence_score: float = Field(ge=0, le=1, description="How well keywords fit together")
    action: str = Field(description="keep, split, or merge")
    notes: str = Field(description="Any concerns or recommendations")


class ClusterAgentInput(BaseModel):
    """Input for cluster agent."""

    clusters: list[dict]  # Each has 'keywords' list and optional metrics
    context: str = Field(default="", description="Optional brand/niche context")


class ClusterAgentOutput(BaseModel):
    """Output from cluster agent."""

    validations: list[ClusterValidation]


class ClusterAgent(BaseAgent[ClusterAgentInput, ClusterAgentOutput]):
    """Agent for validating and naming keyword clusters.

    Used in Step 6 after HDBSCAN clustering to:
    - Validate cluster coherence (do all keywords belong?)
    - Name each cluster descriptively
    - Identify the best primary keyword
    - Flag clusters that should be split or merged
    """

    model = "openai:gpt-4-turbo"
    temperature = 0.4

    @property
    def system_prompt(self) -> str:
        return """You are a keyword clustering expert. Given groups of semantically similar keywords:

1. **Validate Coherence**:
   - Do all keywords in the cluster share the same search intent?
   - Would a single piece of content satisfy all these keywords?
   - If not, flag the cluster for splitting

2. **Name Each Cluster**:
   - Create a descriptive name (2-5 words)
   - Should clearly indicate what the cluster is about
   - Examples: "CRM Comparison Guides", "Email Marketing Tutorials", "Pricing Questions"

3. **Select Primary Keyword**:
   - Choose the keyword with highest search value potential
   - Should be specific enough to rank for
   - Should represent the cluster well

4. **Scoring**:
   - coherence_score 0.9-1.0: Perfect cluster, all keywords highly related
   - coherence_score 0.7-0.9: Good cluster, minor variations acceptable
   - coherence_score 0.5-0.7: Borderline, may need review
   - coherence_score <0.5: Should split or re-cluster

5. **Actions**:
   - keep: Cluster is good as-is
   - split: Cluster contains multiple intents, should be divided
   - merge: Cluster is too narrow, could combine with another

Be precise and practical. Focus on whether keywords can be addressed by a single piece of content."""

    @property
    def output_type(self) -> type[ClusterAgentOutput]:
        return ClusterAgentOutput

    def _build_prompt(self, input_data: ClusterAgentInput) -> str:
        clusters_text = []
        for i, cluster in enumerate(input_data.clusters):
            keywords = cluster.get("keywords", [])
            kw_list = "\n    ".join(f"- {kw}" for kw in keywords[:20])  # Limit for context
            intent = cluster.get("dominant_intent", "unknown")
            volume = cluster.get("total_volume", 0)

            clusters_text.append(
                f"Cluster {i}:\n"
                f"  Intent: {intent}\n"
                f"  Total Volume: {volume}\n"
                f"  Keywords ({len(keywords)} total):\n    {kw_list}"
            )

        context_text = ""
        if input_data.context:
            context_text = f"\nBrand/Niche Context:\n{input_data.context}\n"

        return f"""Validate and name these keyword clusters:{context_text}

{chr(10).join(clusters_text)}

For each cluster, provide:
1. A descriptive name (2-5 words)
2. Brief description
3. The best primary keyword
4. Coherence score (0-1)
5. Recommended action (keep/split/merge)
6. Any notes or concerns"""
