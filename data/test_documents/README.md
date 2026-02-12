# Sample Test Documents

This directory contains sample clinical documents for testing the MedArchive RAG ingestion pipeline.

## Available Documents

### 1. hypertension_guideline.md
A comprehensive clinical practice guideline for hypertension management. This document includes:
- Diagnostic criteria and blood pressure classifications
- Treatment algorithms with medication tables
- Lifestyle modification recommendations
- Special population considerations
- Monitoring and follow-up protocols

**Use for testing:**
- Complex document structure with multiple heading levels
- Tables with medication dosing information
- Mixed content (text, tables, lists)
- Clinical terminology and abbreviations

### 2. diabetes_summary.md
A simplified diabetes management summary document. This document includes:
- Basic treatment protocol
- Medication dosing table
- Monitoring guidelines
- Patient education key points

**Use for testing:**
- Simple document structure
- Basic table parsing
- Short content chunks
- Quick pipeline validation

## Testing Guidelines

### Quick Test
Use `diabetes_summary.md` for:
- Initial pipeline validation
- Quick iteration during development
- Testing individual components

### Comprehensive Test
Use `hypertension_guideline.md` for:
- End-to-end pipeline testing
- Table parsing validation
- Semantic chunking verification
- Real-world document complexity

## Creating Test PDFs

If you need PDF versions for testing:

```bash
# Install pandoc (if not already installed)
# Convert markdown to PDF
pandoc hypertension_guideline.md -o hypertension_guideline.pdf
pandoc diabetes_summary.md -o diabetes_summary.pdf
```

Or use online converters:
- https://www.markdowntopdf.com/
- https://md2pdf.netlify.app/

## Expected Outcomes

### Chunk Count Estimates
- `diabetes_summary.md`: 3-5 chunks
- `hypertension_guideline.md`: 25-35 chunks

### Section Paths Examples
From hypertension_guideline.md:
- `Executive Summary`
- `1. Introduction > 1.1 Background`
- `6. Pharmacological Treatment > 6.1 First-Line Agents > ACE Inhibitors`
- `Appendix A: Drug Interaction Checklist`

### Semantic Search Test Queries
Try these queries after indexing:
1. "What are the first-line medications for hypertension?"
2. "Blood pressure targets for elderly patients"
3. "ACE inhibitor side effects and contraindications"
4. "Lifestyle modifications for hypertension"
5. "Monitoring frequency for patients on diuretics"
