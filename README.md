This repo provides the codebase to reproduce the results and Figures from the paper *"Decomposing drivers of global temperature change after net zero."*
<https://iopscience.iop.org/article/10.1088/1748-9326/aebc9e/meta>

Annika Högner, Zebedee Nicholls, Gaurav Ganti (2026)

In order to reproduce the results from beginning to end you need to:
- request the SCI-2025_v1.1_pathways_ensemble_global.xlsx from https://scenariocompass.org/scenario-dashboard
- create an .env file in the main folder setting the paths as shown in .env.sample
- run uv sync
- start jupyter with uv run jupyter lab
- run the notebooks in the "analysis" folder in order
