"""GroundCheck + LangChain integration example.

Demonstrates how to use GroundCheck as a post-generation verification
step in a LangChain pipeline. GroundCheck verifies that the LLM's
response is grounded in the retrieved context — catching hallucinations
before they reach the user.

Requirements:
    pip install groundcheck langchain langchain-openai

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/langchain_grounded_chain.py
"""

from typing import List

from groundcheck import GroundCheck, Memory

# -- GroundCheck verifier (zero-dependency core, no GPU needed) --
verifier = GroundCheck(neural=False)


def verify_response(
    response: str,
    context_docs: List[str],
    mode: str = "strict",
) -> dict:
    """Verify an LLM response against the retrieved context documents.

    Args:
        response: The LLM-generated answer text.
        context_docs: The source documents used as context.
        mode: "strict" rewrites hallucinations; "permissive" reports only.

    Returns:
        Dict with verification results including pass/fail, corrections,
        hallucinations, and confidence score.
    """
    # Convert context documents to GroundCheck memories
    memories = [
        Memory(id=f"doc-{i}", text=doc, trust=0.9)
        for i, doc in enumerate(context_docs)
    ]

    report = verifier.verify(response, memories, mode=mode)

    return {
        "original": response,
        "passed": report.passed,
        "corrected": report.corrected,
        "hallucinations": report.hallucinations,
        "confidence": report.confidence,
        "contradictions": [
            {
                "slot": c.slot,
                "values": c.values,
                "most_trusted_value": c.most_trusted_value,
            }
            for c in report.contradiction_details
        ],
    }


# ═══════════════════════════════════════════════════════════════
#  Example 1: Standalone verification (no LLM needed)
# ═══════════════════════════════════════════════════════════════

def example_standalone():
    """Verify a response against known facts — no API key needed."""
    print("=" * 60)
    print("Example 1: Standalone verification")
    print("=" * 60)

    context = [
        "The company was founded in 2019 by Alice Chen.",
        "Headquarters are in Austin, Texas.",
        "The company uses PostgreSQL and Redis for its data layer.",
    ]

    # Simulated LLM response with a hallucination
    response = (
        "The company was founded in 2020 by Alice Chen. "
        "They are based in Austin, Texas and use MongoDB for their database."
    )

    result = verify_response(response, context)

    print(f"Passed: {result['passed']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Hallucinations: {result['hallucinations']}")
    if result['corrected']:
        print(f"Corrected: {result['corrected']}")
    print()


# ═══════════════════════════════════════════════════════════════
#  Example 2: LangChain RAG chain with GroundCheck
# ═══════════════════════════════════════════════════════════════

def example_langchain_rag():
    """Full LangChain RAG pipeline with GroundCheck verification.

    Requires: OPENAI_API_KEY environment variable.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        print("Skipping LangChain example (langchain not installed).")
        print("Install with: pip install langchain langchain-openai")
        return

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping LangChain example (OPENAI_API_KEY not set).")
        return

    print("=" * 60)
    print("Example 2: LangChain RAG + GroundCheck")
    print("=" * 60)

    # Simulated retrieval step — in production this would be a vector store
    retrieved_docs = [
        "Nick is a freelance full-stack developer based in Wisconsin.",
        "He primarily works with React, Python, and Node.js.",
        "His current project is CRT-GroundCheck-SSE, a trust-weighted memory system.",
    ]

    # Build the chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based ONLY on the provided context.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # Generate
    context_text = "\n".join(retrieved_docs)
    question = "What does Nick do and where is he located?"
    response = chain.invoke({"context": context_text, "question": question})

    print(f"LLM response: {response}")
    print()

    # Verify with GroundCheck
    result = verify_response(response, retrieved_docs)

    print(f"Passed: {result['passed']}")
    print(f"Confidence: {result['confidence']:.2f}")
    if result['hallucinations']:
        print(f"Hallucinations caught: {result['hallucinations']}")
    if result['corrected']:
        print(f"Corrected response: {result['corrected']}")
    print()


# ═══════════════════════════════════════════════════════════════
#  Example 3: Custom LangChain Tool
# ═══════════════════════════════════════════════════════════════

def example_langchain_tool():
    """Register GroundCheck as a LangChain tool for use in agents."""
    try:
        from langchain_core.tools import tool as langchain_tool
    except ImportError:
        print("Skipping tool example (langchain-core not installed).")
        return

    print("=" * 60)
    print("Example 3: GroundCheck as LangChain Tool")
    print("=" * 60)

    @langchain_tool
    def groundcheck_verify(text: str, facts: str) -> str:
        """Verify that text is grounded in the given facts.
        
        Args:
            text: The text to verify.
            facts: Pipe-separated list of known facts.
        """
        fact_list = [f.strip() for f in facts.split("|") if f.strip()]
        result = verify_response(text, fact_list)
        if result["passed"]:
            return f"VERIFIED (confidence: {result['confidence']:.0%})"
        else:
            corrections = result["corrected"] or "No auto-correction available"
            return (
                f"HALLUCINATION DETECTED: {result['hallucinations']}. "
                f"Corrected: {corrections}"
            )

    # Demo the tool
    output = groundcheck_verify.invoke({
        "text": "User works at Amazon",
        "facts": "User works at Microsoft | User lives in Seattle",
    })
    print(f"Tool output: {output}")
    print()


if __name__ == "__main__":
    example_standalone()
    example_langchain_tool()
    example_langchain_rag()
