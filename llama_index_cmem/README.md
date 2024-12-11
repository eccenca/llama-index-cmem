## CMEMQueryGenerator
**Input:**
- Natural language query or question (NLQuery)
- Ontology graph
- Context graph
- Prompt (optional, overwrites default)

**Output:** Generated SPARQL query to answer the question as [CMEMResult](#CMEMResult)

**Process:** Question + Ontology -> Prompt -> LLM -> SPARQL Query

## CMEMQueryEngine
**Input:**
- Either (generated) SPARQL query  (from [CMEMQueryGenerator](#CMEMQueryGenerator))
- Or existing query from query catalog
- Prompt (optional, overwrites default)
- NLQuery

**Output:** Answer to the question as [CMEMResult](#CMEMResult)

**Process:** SPARQL query -> CMEM -> CMEM context + NLQuery + Prompt -> LLM -> Answer

## CMEMGraphIndex
- embed existing graph
- generate triples from text
- generate ontology

## CMEMResult
```
{
   "question": "My question?",
   "sparql_query": "SELECT * FROM context_graph WHERE ...",
   "json_response": "{"content": "from cmem"}",
   "answer": "My answer!"
}
```
