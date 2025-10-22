"""
Hallucination detection for framework outputs.
Checks grounding, factual consistency, and unsupported claims.
"""

import re
from typing import Dict, List, Any
from collections import Counter


class HallucinationChecker:
    """Check for hallucinations and grounding in outputs."""
    
    def __init__(self):
        """Initialize hallucination checker."""
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.entity_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        
    def check_grounding(self, output: str, context: str) -> Dict[str, Any]:
        """Check if output is grounded in context."""
        # Extract key elements from both
        output_numbers = set(self.number_pattern.findall(output))
        context_numbers = set(self.number_pattern.findall(context))
        
        output_entities = set(self.entity_pattern.findall(output))
        context_entities = set(self.entity_pattern.findall(context))
        
        # Calculate grounding scores
        numbers_grounded = len(output_numbers & context_numbers) / len(output_numbers) if output_numbers else 1.0
        entities_grounded = len(output_entities & context_entities) / len(output_entities) if output_entities else 1.0
        
        # Find unsupported claims
        unsupported_numbers = list(output_numbers - context_numbers)
        unsupported_entities = list(output_entities - context_entities)
        
        # Check for contradictions
        contradictions = self._find_contradictions(output, context)
        
        # Overall grounding score
        grounding_score = (numbers_grounded + entities_grounded) / 2
        
        return {
            'grounding_score': round(grounding_score * 100, 1),
            'numbers_grounded': round(numbers_grounded * 100, 1),
            'entities_grounded': round(entities_grounded * 100, 1),
            'unsupported_numbers': unsupported_numbers[:5],  # Top 5
            'unsupported_entities': unsupported_entities[:5],  # Top 5
            'contradictions': contradictions,
            'factual_consistency': 100 - (len(contradictions) * 20)  # Penalty for contradictions
        }
    
    def _find_contradictions(self, output: str, context: str) -> List[str]:
        """Find contradictory statements between output and context."""
        contradictions = []
        
        # Check for negation mismatches
        output_lower = output.lower()
        context_lower = context.lower()
        
        # Simple contradiction patterns
        if 'not' in output_lower and 'not' not in context_lower:
            if any(word in both for word in ['can', 'will', 'should'] 
                   for both in [output_lower, context_lower]):
                contradictions.append("Negation mismatch detected")
        
        # Check for conflicting numbers (e.g., "5 items" vs "3 items")
        output_quantities = re.findall(r'(\d+)\s+(\w+)', output)
        context_quantities = re.findall(r'(\d+)\s+(\w+)', context)
        
        context_dict = {item: num for num, item in context_quantities}
        for num, item in output_quantities:
            if item in context_dict and num != context_dict[item]:
                contradictions.append(f"Quantity mismatch: {num} {item} vs {context_dict[item]} {item}")
        
        return contradictions[:3]  # Return top 3 contradictions
    
    def check_confidence(self, output: str) -> Dict[str, Any]:
        """Check confidence indicators in output."""
        # Hedging phrases that indicate uncertainty
        hedging_phrases = [
            'maybe', 'perhaps', 'possibly', 'might', 'could be',
            'seems like', 'appears to', 'probably', 'likely',
            'I think', 'I believe', 'not sure', 'uncertain'
        ]
        
        # Confident phrases
        confident_phrases = [
            'definitely', 'certainly', 'absolutely', 'clearly',
            'obviously', 'undoubtedly', 'without a doubt', 'for sure'
        ]
        
        output_lower = output.lower()
        hedging_count = sum(1 for phrase in hedging_phrases if phrase in output_lower)
        confident_count = sum(1 for phrase in confident_phrases if phrase in output_lower)
        
        # Calculate confidence score
        if hedging_count + confident_count == 0:
            confidence_score = 50  # Neutral
        else:
            confidence_score = (confident_count / (hedging_count + confident_count)) * 100
        
        return {
            'confidence_score': round(confidence_score, 1),
            'hedging_phrases_used': hedging_count,
            'confident_phrases_used': confident_count,
            'appropriate_confidence': 40 <= confidence_score <= 80  # Not too low, not too high
        }
