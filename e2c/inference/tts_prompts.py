LLM_COMBINATION_PROMPT = """Role: You are an expert mathematical reasoner and an impartial judge. Your task is to evaluate several proposed plans for solving a given math problem and identify the single best one.
Input:
• Problem: {problem}
• Candidate Plans: A numbered list of K exploration plans.
{explorations}
Instructions:
1. Carefully analyze the problem and each of the K candidate plans.
2. Assess the plans based on their logical soundness, potential for success, and efficiency.
3. Select the single best plan that is most likely to lead to a correct and complete solution.
Output Format: Output only the full text of the single best plan you have selected. Do not add any extra commentary, explanation, or formatting."""

EXPLORATION_PROMPT = """Role: You are a careful math problem solver.
Input:
• Problem: {problem}
Instructions:
• Produce exactly one short reasoning sketch (an exploration) that helps approach the problem. The exploration must be concise (about 2-4 short sentences).
• Do not produce the final answer in this call.
• Stop immediately after the single exploration text and do not append any extra commentary, labels, or formatting.
Output format: A single short exploration paragraph (2-4 short sentences) and nothing else."""