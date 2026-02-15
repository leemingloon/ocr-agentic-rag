"""
Prompt Version Data Class

Represents a single version of a prompt template.
"""

from dataclasses import dataclass


@dataclass
class PromptVersion:
    """
    Prompt version metadata
    
    Attributes:
        id: Database ID
        name: Template name
        template: Prompt template text
        version: Version string (e.g., "2.1")
        approved_by: Email of approver
        approved_at: Timestamp of approval
        status: 'draft', 'approved', or 'deprecated'
    """
    id: int
    name: str
    template: str
    version: str
    approved_by: str
    approved_at: str
    status: str
    
    def __str__(self):
        """String representation"""
        return f"PromptVersion({self.name} v{self.version}, status={self.status})"
    
    def is_active(self) -> bool:
        """Check if version is active (approved)"""
        return self.status == "approved"
    
    def is_deprecated(self) -> bool:
        """Check if version is deprecated"""
        return self.status == "deprecated"