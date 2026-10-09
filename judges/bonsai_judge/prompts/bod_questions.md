You are designing a grading rubric for answers to a user's information need.

<information_need>
<%=topic_query%>
</information_need>

Write the questions a careful expert would use to grade how well an answer meets this information need. Each question checks one distinct part of what the user needs to know or do.

Rules:
- Write between <%=min_questions%> and <%=max_questions%> questions. Use as many as the need has genuinely distinct parts: a narrow, single-fact need gets few questions; a broad, multi-part need gets more. Do not pad the list to reach a number.
- Each question must be gradable on a scale from "not addressed" to "fully answered with specific detail", so phrase it as "Does the answer ...?" about content, e.g. "Does the answer explain why ...?" or "Does the answer give ...?".
- Every question must check substantive content: a fact, explanation, step, or piece of advice the user asked for or clearly needs to act on what they asked.
- Use any background or situation described only to decide WHAT content the user needs (for example, a safety concern makes safety advice necessary). Do not write questions that merely check whether the answer acknowledges, mentions, or relates to the user's interests, hobbies, or feelings.
- No two questions may check the same thing. Do not ask about tone, reading level, style, length, formatting, or citations.
- Do not mention any particular source, product, or answer.

Return ONLY a JSON array of strings, one per question, with no other text.
