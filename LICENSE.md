# SimBench Licensing Information

This document provides detailed information about the licensing structure for SimBench, which follows a multi-level licensing approach to ensure appropriate attribution and compliance with all source dataset requirements.

## Overview

SimBench employs a **multi-level licensing structure** that distinguishes between:
1. The SimBench framework (code, pipeline, and methodology)
2. The 20 constituent datasets that comprise the benchmark

This approach ensures compliance with all original dataset terms while providing clarity for users about what permissions apply to different components of the benchmark.

## Framework License

**The SimBench framework** (including all code, pipeline, documentation, and methodology) is licensed under:

**CC-BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)**

This means you are free to:
- Share: copy and redistribute the framework in any medium or format
- Adapt: remix, transform, and build upon the framework

Under the following terms:
- **Attribution**: You must give appropriate credit and indicate if changes were made
- **NonCommercial**: You may not use the material for commercial purposes
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license

## Constituent Dataset Licenses

SimBench incorporates 20 datasets from diverse sources, each with its own licensing terms:

### Explicitly Licensed Datasets (17/20)
The majority of constituent datasets (17 out of 20) are governed by explicit permissive licenses, including:
- Creative Commons licenses (various types)
- MIT License
- Other open source licenses

### Publicly Available Datasets (3/20)
A small number of datasets (3 out of 20) are publicly available for research purposes but do not have explicit open source licenses. For these datasets, SimBench's inclusion is based on the principle of **transformative use**.

## Transformative Use Principles

SimBench qualifies as transformative use of the constituent datasets because:

1. **No Raw Data Distribution**: SimBench does not contain or redistribute any raw, individual-level participant data from the original datasets.

2. **Derivative Work**: SimBench represents a new, derivative work consisting entirely of:
   - Aggregated group-level response distributions
   - Non-reversible statistical summaries
   - Unified formatting and structure

3. **Privacy Protection**: The aggregation process protects the privacy of original human subjects by making individual responses non-identifiable and non-recoverable.

4. **New Purpose**: The benchmark serves a fundamentally different purpose (evaluating LLM simulation capabilities) than the original datasets (studying specific social/behavioral phenomena).

## Usage Guidelines

### For Framework Components
Users may freely use, modify, and distribute the SimBench framework code and methodology under the CC-BY-NC-SA 4.0 license terms.

### For Dataset Content
When using SimBench's aggregated data distributions:
1. Cite both SimBench and the relevant original dataset sources
2. Acknowledge that the data represents aggregated, derivative distributions
3. Consult original dataset documentation for any additional attribution requirements
4. Respect any usage restrictions from original dataset licenses

### Commercial Use
- The SimBench framework is licensed for non-commercial use only
- Commercial applications must contact the authors for alternative licensing arrangements
- Users should verify commercial use permissions for any constituent datasets they plan to use

## Dataset-Specific Licensing Details

Below is the complete listing of all 20 constituent datasets, their original sources, and specific license terms as promised in the ethics statement:

### Datasets with explicit licenses or terms (per paper: 17/20)

#### Creative Commons Licensed Datasets
1. **ESS (European Social Survey)** - *Creative Commons*
2. **AfroBarometer** - *Creative Commons* 
3. **OSPsychBig5** - *Creative Commons*
4. **OSPsychMGKT** - *Creative Commons*
5. **OSPsychMACH** - *Creative Commons*
6. **OSPsychRWAS** - *Creative Commons*

#### CC-BY-NC-SA 4.0 Licensed Datasets
7. **GlobalOpinionQA** - *CC BY-NC-SA 4.0*
   - Source: World Values Survey and Pew Global Attitudes Survey
8. **DICES-990** - *CC BY-NC-SA 4.0*  
   - Source: Diversity in Conversational AI Evaluation for Safety

#### CC0 (Public Domain) Licensed Datasets
9. **NumberGame** - *CC0 1.0*
10. **ConspiracyCorr** - *CC0 1.0 Universal*

#### MIT Licensed Datasets
11. **WisdomOfCrowds** - *MIT License*

#### Academic/Research-Only Licensed Datasets
12. **Jester** - *"Freely available for research use when cited appropriately"*
13. **Choices13k** - *"All data are available to the public without registration"*
    - Available at: github.com/jcpeterson/choices13k
14. **ISSP (International Social Survey Programme)** - *"Data and documents are released for academic research and teaching"*
    - Source: Cross-national collaborative surveys (2017-2021)

### Datasets without explicit licenses — Transformative Use Basis (3/20)

#### Publicly Available Research Datasets
15. **OpinionQA** - *No licensing information provided*
    - Source: Pew Research American Trends Panel
    - Status: Publicly available research dataset without explicit license
    - Basis: Transformative use - aggregated distributions only

16. **ChaosNLI** - *No licensing information provided*  
    - Status: Freely available without registration
    - Available at: CodaLab worksheets platform
    - Basis: Transformative use - aggregated distributions only

17. **MoralMachineClassic** - *No licensing information provided*
    - Status: Publicly available research dataset
    - Basis: Transformative use - aggregated distributions only

#### Research-Permitted Datasets
18. **LatinoBarómetro** - *No explicit language forbidding redistribution*
    - Source: Annual public opinion survey across Latin America
    - Basis: Public research use, transformative aggregation

19. **MoralMachine** - *No formal open license declared, but authors explicitly state dataset may be used beyond replication for follow-up research questions*
    - Source: MIT Moral Machine experiment  
    - Basis: Explicit research permission + transformative use

20. **TISP (Trust in Science and Science-Related Populism)** - *No explicit language forbidding redistribution*
    - Source: Multi-country science perception survey
    - Basis: Research dataset, transformative aggregation

## Legal Basis for Inclusion

### For Explicitly Licensed Datasets (17/20)
These datasets are included under their respective open source or academic licenses, with full attribution and compliance with stated terms.

### For Non-Explicitly Licensed Datasets (3/20)
These datasets are included based on **transformative use** principles:

1. **Public Availability**: All are publicly available research datasets
2. **Academic Purpose**: Original datasets were created for academic research
3. **Transformative Nature**: SimBench contains only non-reversible, aggregated statistical distributions
4. **Privacy Protection**: No individual responses or personally identifiable information
5. **Different Purpose**: Evaluation of LLM simulation capabilities (not original research purpose)
6. **Attribution**: Full citation and acknowledgment of original sources

## Usage Requirements by Category

### Creative Commons Datasets
- **Share-Alike**: Derivative works must maintain compatible licensing where applicable
- **Non-Commercial**: Some restrictions apply (see individual CC license variants)

### MIT Licensed Datasets  
- **Commercial Use**: Generally permitted with attribution

### CC0/Public Domain Datasets
- **No Restrictions**: May be used freely
 - **Attribution Recommended**: While not required, citation is encouraged

### Academic Research Licensed Datasets
- **Academic Use**: Restricted to educational and research purposes
- **Commercial Use**: Generally prohibited or requires separate permission

### Transformative Use Datasets
- **Research Use**: Should be limited to academic and research contexts
- **Aggregated Use Only**: Use only the aggregated distributions, not for individual data recovery

## Questions and Clarifications

If any licensing terms are unclear, or if you require permissions beyond those stated here, please consult the original dataset owners or data custodians first; their terms govern. For questions about the SimBench framework licensing, please reach out through the appropriate review channels for this submission.

---

*This licensing structure ensures that SimBench can be used responsibly while respecting the terms and intentions of all contributing dataset creators and maintaining transparency about data provenance and usage rights.*