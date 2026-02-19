"""SQLAlchemy database models."""
from dotenv import load_dotenv
from app.models.base import Base
from app.models.brand import BrandProfile
from app.models.content import (
    ContentArticle,
    ContentArticleVersion,
    ContentBrief,
    WriterInstructions,
)
from app.models.discovery_snapshot import DiscoveryTopicSnapshot
from app.models.discovery_learning import DiscoveryIterationLearning
from app.models.keyword import Keyword, SeedTopic
from app.models.oauth_account import OAuthAccount
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project
from app.models.style_guide import BriefDelta, ProjectStyleGuide
from app.models.topic import Topic
from app.models.user import User


load_dotenv()

__all__ = [
    "Base",
    "User",
    "OAuthAccount",
    "Project",
    "BrandProfile",
    "Keyword",
    "SeedTopic",
    "Topic",
    "ContentBrief",
    "WriterInstructions",
    "ContentArticle",
    "ContentArticleVersion",
    "ProjectStyleGuide",
    "BriefDelta",
    "PipelineRun",
    "StepExecution",
    "DiscoveryTopicSnapshot",
    "DiscoveryIterationLearning",
]
