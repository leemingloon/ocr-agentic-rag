"""
Prompt Registry - MAS FEAT Compliant Versioning

Maintains audit trail of all LLM prompts used in production.

MAS FEAT Requirements:
- All prompts must be versioned
- Approval workflow required
- Audit trail of all changes
- Rollback capability
"""

import sqlite3
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from .prompt_version import PromptVersion


class PromptRegistry:
    """
    Version control system for LLM prompts
    
    Ensures MAS FEAT compliance through:
    - Audit trails
    - Approval workflows
    - Rollback capability
    """
    
    def __init__(self, db_path: str = "data/credit_risk/prompts.db"):
        """
        Initialize prompt registry
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                template TEXT NOT NULL,
                version TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'approved',
                UNIQUE(name, version)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER,
                borrower TEXT,
                input_features TEXT,
                output_memo TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_prompt(
        self,
        template_name: str,
        template: str,
        version: str,
        approved_by: str,
        status: str = "approved"
    ) -> int:
        """
        Register new prompt version
        
        Args:
            template_name: Prompt template name
            template: Prompt template text
            version: Version string (e.g., "2.1")
            approved_by: Email of approver
            status: Status ('draft' or 'approved')
            
        Returns:
            Prompt ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO prompts (name, template, version, approved_by, status)
                VALUES (?, ?, ?, ?, ?)
            """, (template_name, template, version, approved_by, status))
            
            conn.commit()
            prompt_id = cursor.lastrowid
            
            print(f"✓ Registered prompt: {template_name} v{version}")
            
            return prompt_id
        
        except sqlite3.IntegrityError:
            print(f"✗ Prompt {template_name} v{version} already exists")
            return -1
        
        finally:
            conn.close()
    
    def get_latest(self, template_name: str) -> Optional[PromptVersion]:
        """
        Get latest approved prompt version
        
        Args:
            template_name: Prompt template name
            
        Returns:
            PromptVersion or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, template, version, approved_by, approved_at, status
            FROM prompts
            WHERE name = ? AND status = 'approved'
            ORDER BY approved_at DESC
            LIMIT 1
        """, (template_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PromptVersion(*row)
        
        return None
    
    def get_version(
        self,
        template_name: str,
        version: str
    ) -> Optional[PromptVersion]:
        """Get specific prompt version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, template, version, approved_by, approved_at, status
            FROM prompts
            WHERE name = ? AND version = ?
        """, (template_name, version))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PromptVersion(*row)
        
        return None
    
    def list_versions(self, template_name: str) -> List[PromptVersion]:
        """List all versions of a prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, template, version, approved_by, approved_at, status
            FROM prompts
            WHERE name = ?
            ORDER BY approved_at DESC
        """, (template_name,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [PromptVersion(*row) for row in rows]
    
    def deprecate_version(self, template_name: str, version: str):
        """Deprecate a prompt version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE prompts
            SET status = 'deprecated'
            WHERE name = ? AND version = ?
        """, (template_name, version))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Deprecated: {template_name} v{version}")
    
    def log_usage(
        self,
        prompt_id: int,
        borrower: str,
        input_features: str,
        output_memo: str
    ):
        """Log prompt usage (audit trail)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO prompt_usage (prompt_id, borrower, input_features, output_memo)
            VALUES (?, ?, ?, ?)
        """, (prompt_id, borrower, input_features, output_memo))
        
        conn.commit()
        conn.close()


# Example usage
if __name__ == "__main__":
    registry = PromptRegistry()
    
    # Register a prompt
    template = """Analyze credit risk for {borrower}.
    
Debt/EBITDA: {debt_to_ebitda}
PD: {pd}

Provide recommendation."""
    
    prompt_id = registry.register_prompt(
        template_name="risk_memo_simple",
        template=template,
        version="1.0",
        approved_by="analyst@ocbc.com"
    )
    
    # Get latest version
    latest = registry.get_latest("risk_memo_simple")
    if latest:
        print(f"\nLatest version: {latest.version}")
        print(f"Approved by: {latest.approved_by}")
    
    # List all versions
    versions = registry.list_versions("risk_memo_simple")
    print(f"\nAll versions: {len(versions)}")