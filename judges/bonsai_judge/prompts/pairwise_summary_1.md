# Pairwise Summary Comparison

You are a capable and knowledgeable subject matter expert, you need to select the better of two RAG summaries (A or B) that satisfy the information needs given in a query.

The query used to find the relevant documents and generate the summary is "<%=topic_query%>". Choose between the two presented summaries which satisfies the information needs better.

## Summary A

<%=summary_a%>

## Summary B

<%=summary_b%>

## Your decision

Judge which summary better satisfies the information need in the query, considering relevance, completeness, and how well the claims are grounded. Respond with exactly one character and nothing else: `A` if Summary A is better, or `B` if Summary B is better. No explanation, punctuation, or whitespace.
