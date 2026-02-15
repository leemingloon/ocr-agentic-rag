"""
Safety Filter - LLM Output Validation

Ensures LLM outputs comply with regulatory and policy requirements.

Filters:
1. Banned phrases (overconfident predictions)
2. Sensitive information leakage
3. Non-compliant language

MAS FEAT Compliance: Prevent misleading or overconfident statements.

Usage:
    filter = SafetyFilter()
    
    raw_output = "ABC Corp will definitely default within 6 months."
    filtered = filter.filter(raw_output)
    # Returns: "ABC Corp may face elevated default risk..."
"""

from typing import List, Dict
import re


class SafetyFilter:
    """
    Filter LLM outputs for compliance
    
    Prevents:
    - Overconfident predictions
    - Banned phrases
    - Sensitive data leakage
    """
    
    def __init__(self):
        """Initialize safety filter"""
        # Banned phrases (regulatory compliance)
        self.banned_phrases = [
            "guaranteed default",
            "certain bankruptcy",
            "zero risk",
            "100% probability",
            "will definitely default",
            "cannot possibly default",
            "risk-free",
            "absolutely certain",
        ]
        
        # Sensitive patterns to redact
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
        ]
        
        # Overconfident terms to hedge
        self.overconfident_terms = {
            "will default": "may default",
            "certain": "likely",
            "definitely": "probably",
            "impossible": "unlikely",
            "guaranteed": "expected",
            "always": "typically",
            "never": "rarely",
        }
    
    def filter(self, text: str) -> str:
        """
        Filter LLM output
        
        Args:
            text: Raw LLM output
            
        Returns:
            Filtered text
        """
        filtered = text
        
        # Step 1: Remove banned phrases
        for phrase in self.banned_phrases:
            if phrase.lower() in filtered.lower():
                # Replace entire sentence
                filtered = self._remove_sentence_with_phrase(filtered, phrase)
                filtered += f"\n\n[FILTERED: Overconfident statement removed]"
        
        # Step 2: Redact sensitive information
        for pattern in self.sensitive_patterns:
            filtered = re.sub(pattern, "[REDACTED]", filtered, flags=re.IGNORECASE)
        
        # Step 3: Hedge overconfident terms
        filtered = self._add_hedging(filtered)
        
        # Step 4: Add disclaimer if not present
        if "recommendation" in filtered.lower() and "should be reviewed" not in filtered.lower():
            filtered += "\n\n*All recommendations should be reviewed by a qualified credit analyst.*"
        
        return filtered
    
    def _remove_sentence_with_phrase(self, text: str, phrase: str) -> str:
        """Remove sentence containing banned phrase"""
        sentences = text.split('.')
        filtered_sentences = [
            s for s in sentences
            if phrase.lower() not in s.lower()
        ]
        return '.'.join(filtered_sentences)
    
    def _add_hedging(self, text: str) -> str:
        """Add appropriate hedging language"""
        filtered = text
        
        for overconfident, hedged in self.overconfident_terms.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(overconfident), re.IGNORECASE)
            filtered = pattern.sub(hedged, filtered)
        
        return filtered
    
    def validate(self, text: str) -> Dict:
        """
        Validate text without filtering
        
        Returns:
            {
                "is_compliant": bool,
                "violations": List[str],
            }
        """
        violations = []
        
        # Check banned phrases
        for phrase in self.banned_phrases:
            if phrase.lower() in text.lower():
                violations.append(f"Banned phrase: '{phrase}'")
        
        # Check sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Sensitive information detected")
        
        # Check overconfident language
        for term in self.overconfident_terms.keys():
            if term.lower() in text.lower():
                violations.append(f"Overconfident language: '{term}'")
        
        return {
            "is_compliant": len(violations) == 0,
            "violations": violations,
        }


# Example usage
if __name__ == "__main__":
    filter = SafetyFilter()
    
    # Example 1: Banned phrase
    raw1 = "ABC Corp will definitely default within 6 months. This is a certain bankruptcy."
    filtered1 = filter.filter(raw1)
    print("Example 1:")
    print(f"Raw: {raw1}")
    print(f"Filtered: {filtered1}")
    
    # Example 2: Overconfident language
    raw2 = "The company will always meet its obligations and can never default."
    filtered2 = filter.filter(raw2)
    print("\nExample 2:")
    print(f"Raw: {raw2}")
    print(f"Filtered: {filtered2}")
    
    # Example 3: Validation
    validation = filter.validate("This investment is guaranteed to be risk-free.")
    print("\nExample 3 - Validation:")
    print(f"Compliant: {validation['is_compliant']}")
    print(f"Violations: {validation['violations']}")