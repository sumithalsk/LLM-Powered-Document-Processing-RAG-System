"""
LLM integration with OpenAI GPT models and prompt optimization.
"""
from typing import Optional, Dict, Any
import openai
from openai import OpenAI


class LLMHandler:
    """Handles LLM integration with prompt optimization."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM handler.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create an optimized prompt for RAG.
        
        Args:
            query: User query
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Be concise and accurate in your response.
4. If relevant, cite which source(s) you used from the context.

Answer:"""
        return prompt
    
    def generate_response(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with response and metadata
        """
        # Create optimized prompt
        prompt = self.create_rag_prompt(query, context)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'model': self.model,
                'tokens_used': response.usage.total_tokens,
                'finish_reason': response.choices[0].finish_reason
            }
        
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'model': self.model,
                'error': str(e)
            }
    
    def generate_simple_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate a simple response without RAG context.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
