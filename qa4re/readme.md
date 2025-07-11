#### Vanilla RE Prompt:
```
Given the following passage, and two entities, classify their relationship.

Passage: [passage text]
Entity 1: [entity1 text]
Entity 2: [entity2 text]
Possible relations: [relation type list, e.g., positive_correlation, negative_correlation, ...]
Relationship:
```

#### QA4RE Prompt:

```
Determine which option can be inferred from the following passage.

Passage: [passage text]
Options:
A. [entity1 text] and [entity2 text] have a positive correlation.
B. [entity1 text] and [entity2 text] have a negative correlation.
C. [entity1 text] and [entity2 text] have an association.
D. [entity1 text] and [entity2 text] have no known relation.

Which option can be inferred?
Option:
```


#### Improvements for later:
1. Use LLM instead of mini version
2. Use constrains on the relations to exculed the unreasonable relations (instead of including all the relations in the options)
3. 