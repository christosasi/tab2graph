# An interpretable AI Scientist for Drug Repurposing  
*A tabular-to-graph reasoning layer for multimodal datasets*

---

## TL;DR

This project proposes a **Python package** that converts large, heterogeneous tabular datasets (Parquet/CSV/Pandas/Polars) into **typed, temporal knowledge graphs**, enabling a **reasoning agent (LLM + GNN)** to:

- Run **automated reasoning experiments** (e.g. “What if Drug X is repurposed for Disease Y in patients with mutation Z?”).
- Perform **hypothesis testing** against real-world temporal data (approvals, trial outcomes, safety events).
- Generalize beyond medicine to **any structured dataset** that can be represented as a graph.

The flagship use case that will be tested during the development of this package is an Automated Biomedical Reasoning agent.
Primarily we want to see impact on interpretability of AI models when they are trained on dynamic graphs generated from heterogenous and multimodal datasets to solve link predition and node classification tasks. Therefore the core software layer of this package will strive to remain **domain-agnostic**. 

---

## Motivation

Drug repurposing is inherently **multidisciplinary**: clinicians, geneticists, pharmacologists, and data scientists all work with different data silos:

- Clinical trials, EHR/claims, omics, literature, curated knowledge graphs.
- Ontologies for drugs, genes, diseases, pathways, phenotypes.

Today, these are often handled by **separate tools and pipelines**, making it hard to:

- Pose precise, multi-hop questions (drug → gene → pathway → phenotype → trial outcome).
- Test hypotheses against **temporal** evidence (what was knowable in year T vs what happened later).
- Systematically compare **alternative hypotheses** (e.g. different repurposing candidates).

Regulators (e.g. FDA/EMA) already rely on prior evidence (trials, safety data, post-marketing studies) to evaluate repurposed indications. A computational layer is needed which **mirrors this reasoning process** on multimodal, time-stamped data.

---

## Core Concept

At the core, is a **tabular-to-graph reasoning layer**:

> A Python package that takes large tabular datasets (Parquet/CSV/Pandas/Polars),  
> builds **typed, temporal graphs**,  
> and exposes them through a clean API so an **LLM-based agent** can:
> - formulate hypotheses,
> - request specific subgraphs,
> - run predictive models (e.g. GNNs),
> - and test those hypotheses against historical outcomes.

This layer is the **main software component** required to deploy a medical reasoning agent capable of:

- Data analysis on large ontological datasets.
- Automated reasoning experiments.
- Hypothesis testing with real-world temporal benchmarks.

---

## Architecture Overview

The system can be described in three layers:

1. **Data & Graph Layer (this package)**  
   - Ingests Parquet/CSV/Pandas/Polars from multiple sources.  
   - Maps columns to ontologies (drugs, genes, diseases, trials, patients, etc.).  
   - Builds **typed, temporal graphs** and query-specific subgraphs.

2. **Modeling Layer (GNN + temporal/statistical models)**  
   - Operates on graph objects to perform:
     - Link prediction (e.g. drug–disease, drug–AE).
     - Node classification (e.g. approval likelihood).
     - Temporal forecasting (e.g. safety event risk over time).
   - Uses PyTorch Geometric / DGL / standard ML toolchains.

3. **Agent Layer (LLM with tool calling)**  
   - Parses user prompts (e.g. disease, gene, drug, population).  
   - Plans **graph-building and modeling steps**.  
   - Calls package APIs to construct graphs and run models.  
   - Synthesizes results into **human-readable reports** with explanations and references.

---

## Why Graphs & Why LLM + GNN?

**Why graphs?**

- Many domains (especially biomedicine) are naturally graph-shaped:
  - Drug–target–disease–symptom–trial relationships.
  - Patients connected to treatments, outcomes, and biomarkers.
- Graphs support **multimodal integration**:
  - Numeric features (lab values), categorical (labels), text-derived embeddings, time stamps.
- GNNs can learn **structured patterns**:
  - Repurposing signals, safety profiles, response subtypes, trial success patterns.

**Why combine LLMs and GNNs?**

- **LLMs are good at**:
  - Hypothesis generation from text and prior knowledge.
  - Translating natural language questions into structured queries.
  - Explaining mechanisms and summarizing literature.

- **GNNs / models are good at**:
  - Learning from **graph-encoded data** with clear structure.
  - Making **quantitative predictions** grounded in real-world outcomes.
  - Using temporal information systematically.

Together, they enable **automated reasoning experiments** where:

- The LLM proposes *what to test* and *how to test it*.
- The graph + GNN stack provides *evidence* and *numbers*.
- The agent compares outcomes against history to **accept, refine, or reject hypotheses**.

---

## Proposed Python Package

### Goals

Create a **modular Python package** that:

- Converts tabular data into **typed, temporal graph objects**.
- Encodes **feature engineering recipes** per data type.
- Exposes APIs an LLM can control via tool calling.
- Makes it easy to set up **repeatable reasoning experiments and hypothesis tests**.

### Core Capabilities

#### 1. Flexible Data Ingestion

- Load from **Parquet/CSV/Pandas/Polars**.
- Support multiple ontologies / ID systems (e.g. OpenTargets IDs, ChEMBL IDs, HGNC, disease ontologies).
- Track temporal indices (e.g. `start_date`, `approval_date`, `publication_year`) for **time-aware graphs**.

#### 2. Schema & Ontology Configuration

- Define schemas in YAML/JSON or via LLM-generated templates (validated by the package):
  - **Node types** and ID columns (e.g. `drug_id`, `gene_id`, `disease_id`, `trial_id`, `patient_id`).
  - **Edge types** and relationship columns (e.g. `treats`, `targets`, `associated_with`).
  - **Feature sets** and types (numeric, categorical, text, temporal, fingerprints).

#### 3. Automatic Graph Construction

- Automatically validate:
  - Node columns vs edge columns vs feature columns.
- Build:
  - Node indices for homogeneous and heterogeneous graphs.
  - Edge indices across ontologies.
  - Feature matrices per node/edge type.
- Return a **structured graph object** (e.g. PyG-style) with fields like:
  - `x`, `edge_index`, `edge_attr`, `time`, `metadata`.

**Example sketch:**

```python
from tab2graph import GraphBuilder, GraphConfig

config = GraphConfig(
    node_types={
        "drug":    {"id_col": "drug_id"},
        "gene":    {"id_col": "gene_id"},
        "disease": {"id_col": "disease_id"},
        "trial":   {"id_col": "trial_id"},
    },
    edge_types=[
        ("drug", "treats",  "disease", {"table": "drug_disease"}),
        ("drug", "targets", "gene",    {"table": "drug_gene"}),
        ("trial", "tests",  "drug",    {"table": "trial_drug"}),
    ],
    features={
        "drug":    ["atc_code", "structure_fingerprint"],
        "gene":    ["expression_level", "pathways"],
        "disease": ["phenotype_score"],
        "trial":   ["phase", "n_patients", "endpoint_success"],
    },
)

builder = GraphBuilder(config)
graph = builder.build_from_parquet("/data/opentargets/")
# graph.x, graph.edge_index, graph.edge_attr, graph.time, graph.metadata
````

#### 4. Query-Driven Subgraph Extraction

* LLM/agent generates a query specification (e.g. *disease*, *gene*, *drug*, *time window*, *population*).
* Package returns **subgraphs** matching that specification for:

  * Link prediction (e.g. future drug–disease link),
  * Node classification (e.g. approval likelihood),
  * Temporal prediction (e.g. hazard of safety event).

#### 5. Feature Encoding Automation

To scale across many datasets and ontologies, the package includes a **feature encoding spec**, which can be chosen by the agent but enforced by the library:

```python
feature_encoding = {
    "categorical":        "one_hot",
    "multi_label":        "multi_hot",
    "numeric":            "z_score_normalization",
    "ordinal":            "integer_encoding",
    "binary":             "boolean_cast",
    "high_dim_fingerprint": "dense_projection",
    "text_features":      "embedding_bag",
    "temporal_features":  "time_delta_norm",
}
```

The agent and package together:

* Inspect each feature’s datatype.
* Assign appropriate encoding using a standard library of transforms.
* Produce consistent feature matrices for each node/edge type.
* Perform size/shape checks and logging for reproducibility.

This automation removes much of the **manual data wrangling** usually required before graph learning.

#### 6. Agent Integration

* The package is designed for **tool calling**:

  * `list_datasets()`, `inspect_schema()`, `build_graph(...)`, `extract_subgraph(...)`.
* An LLM agent uses these tools to:

  * Translate user prompts into graph queries.
  * Trigger model runs (e.g. “train link predictor on this subgraph, using years ≤ 2015 as training”).
  * Retrieve predictions and graph slices for explanation.

---

## Flagship Use Case: Drug Repurposing & Trial Emulation

For drug repurposing and clinical trial reasoning, the package connects:

* **Drugs**: identifiers, mechanisms, ATC codes, labels, doses.
* **Targets / Genes**: expression, variants, pathways, protein interactions.
* **Diseases / Phenotypes**: ontologies, symptom clusters, severity scales.
* **Trials**: phases, endpoints, outcomes, eligibility criteria, timelines.
* **Safety events**: adverse events, warnings, withdrawals.
* **Patients / cohorts** (where available): comorbidities, treatments, lab values, outcomes.
* **Literature**: temporal publication graph (trials, case reports, observational studies).

Building a **temporal knowledge graph** over these entities allows us to:

* Emulate trial outcomes from historical data.
* Learn patterns in drug–target–disease space that predict:

  * Approval probability for new indications,
  * Efficacy in specific subgroups,
  * Safety events and toxicity patterns.
* Compare model predictions to **real-world timelines** of approvals, failures, and safety signals.

---

## Example Queries (Reasoning + Graph Access)

The following prompts demonstrate where the tabular-to-graph layer is essential:

* **Drug approval prediction**
  *“Given the historical trials of Drug X and its targets, what is the predicted probability that it will be approved for Disease Y within 5 years? Show the key trials and pathways driving this prediction.”*

* **Repurposing candidates**
  *“List 5 candidate drugs for Disease Y based on shared targets, pathways, and clinical phenotypes, ranked by predicted efficacy and safety. Provide the supporting subgraph and key studies for each candidate.”*

* **Patient-level reasoning**
  *“For a patient with gene expression profile G and comorbidities C, which repurposed drugs are likely to be effective and safe? Explain using similar patients and pathways in the graph.”*

* **Safety signal forecasting**
  *“Based on previous safety events in drug class K and related targets, which adverse events should be prioritized for monitoring if Drug X is repurposed for Disease Y?”*

* **Mechanism-aware explanation**
  *“Explain why Drug X is a plausible candidate for Disease Y in patients with mutation Z, combining targets, pathways, and trial evidence.”*

* **Trial emulation**
  *“Simulate the expected efficacy and safety profile for a Phase II trial of Drug X in Disease Y, using historical data from similar molecules and indications.”*

Each query triggers a **repeatable reasoning experiment**: build subgraphs, run models, compare against held-out temporal data.

---

## Automated Reasoning Experiments & Hypothesis Testing

The package explicitly supports **experiments** like:

1. **Hypothesis definition**

   * LLM formulates:
     *“Drug X is a promising candidate for Disease Y in patients with mutation Z.”*

2. **Experiment setup**

   * Agent:

     * Requests subgraphs restricted to **what was known before year T**.
     * Chooses model(s) (e.g. GNN for link prediction).
     * Defines evaluation horizon (e.g. years T+1 to T+5).

3. **Model run and evaluation**

   * Package builds graph(s) and runs the models.
   * Predictions are compared against actual approvals, trial outcomes, safety events.

4. **Hypothesis refinement**

   * LLM inspects:

     * Errors and false positives/negatives,
     * Important nodes/edges/features.
   * Proposes refined hypotheses (e.g. narrower subgroups, different targets, different comparators).

By repeating this loop, users (and eventually the agent itself) can conduct **systematic, data-backed hypothesis testing** over large biomedical graphs.

---

## Evaluation Plan (Short)

The package and the agent stack can be validated on the following tasks:

* **Link prediction**

  * Task: predict future drug–disease or drug–AE edges that later become supported by trials or labels.
  * Metrics: ROC-AUC, AUPRC, precision@K.

* **Approval / outcome prediction**

  * Task: predict whether a drug will gain a new indication or fail, given evidence up to time T.
  * Metrics: time-to-event discrimination (e.g. C-index), calibration.

* **Safety event forecasting**

  * Task: predict emerging safety signals for drug classes/mechanisms.
  * Metrics: early warning performance at predefined lead times.

* **Trial emulation**

  * Task: emulate historical Phase II/III trials using pre-existing data.
  * Metrics: effect size error, ranking of successful vs failed compounds.

Baselines will include:

* Static KG models without temporal structure.
* Non-graph ML on tabular data.
* RAG-style LLMs without graph-based prediction.

---

## FAIR & Safety

**FAIR principles**

* **Findable**: configuration files, graph recipes, and model cards are versioned and indexed.
* **Accessible**: standard formats (Parquet, Arrow) and documented APIs.
* **Interoperable**: uses community ontologies and exports to common graph ML frameworks.
* **Reusable**: reproducible pipelines and example notebooks for academia and industry.

**Safety & intended use**

* The system is intended as a **research and decision-support tool**, not an automated prescribing engine.
* All outputs are **hypotheses, risk estimates, and explanations** that require expert review and regulatory oversight.

---

## Scope for Collaboration

Collaboration and support from **academic labs, hospitals, and industry partners** who are interested in:

* Co-developing and open-sourcing the **tabular-to-graph package**.
* Integrating it with:

  * Biomedical LLMs,
  * GNN toolkits / benchmarks,
  * Agent frameworks for orchestration.
* Evaluating the stack on:

  * Historical drug repurposing case studies,
  * Clinical trial emulation,
  * Safety signal prediction,
  * Other domains where tabular → graph → reasoning is valuable.

The long-term goal is a reusable, interpretable **reasoning layer** that makes **automated reasoning experiments and hypothesis testing** over large multimodal datasets both practical and trustworthy.

