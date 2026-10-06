This repo provides the codebase to reproduce the results and Figures from the preprint "Decomposing drivers of global temperature change after net zero."

In order to reproduce the results from beginning to end you need to:

    - request the SCI-2025_v1.1_pathways_ensemble_global.xlsx
    - from https://scenariocompass.org/scenario-dashboard
    - create a .env file in the main folder setting the paths as shown in .env.sample
    - unzip the drivers_post_net_zero.zip
    - run uv sync
    - start jupyter with uv run jupyter lab
    - run the notebooks and .py scripts in order

Current most up-to-date version of the workflow sits in PR[https://github.com/gaurav-ganti/netzero_assessment/pull/37]
