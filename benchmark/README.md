# Push-button Forecasting Benchmark

<img width="946" alt="Push-button forecasting" src="https://github.com/ServiceNow/research-starcaster/assets/2374980/c3620848-7eda-46e3-bc11-6ed2d06022e4">


## Setting environment variables

Here is a list of all environment variables which the Context-is-Key benchmark will access:

* Storage configuration
  * `CIK_MODEL_STORE`\
    Store the model weights of the baselines in this folder.\
    Default to `"./models"`
  * `CIK_DATA_STORE`\
    Store the downloaded datasets in this folder.\
    Default to `"benchmark/data"`
  * `CIK_DOMINICK_STORE`\
    Store the downloaded dataset for the tasks using the Dominick dataset in this folder.\
    Default to `CIK_DATA_STORE` + `"/dominicks"`
  * `CIK_TRAFFIC_DATA_STORE`\
    Store the downloaded dataset for the tasks using the Traffic dataset in this folder.\
    Default to `CIK_DATA_STORE` + `"/traffic_data"`
  * `HF_HOME`\
    Location of the cache when downloading some datasets from Hugging Face.\
    Default to `CIK_DATA_STORE` + `"/hf_cache"`
* Evaluation configuration
  * `CIK_RESULT_CACHE`\
    Store the output of the baselines in this folder, to avoid recomputing them.\
    Default to `"./inference_cache"`
  * `CIK_METRIC_SCALING_CACHE`\
    Store the scaling factors for each task in this folder, to avoid recomputing them.\
    Default to `"./metric_scaling_cache"`
  * `CIK_METRIC_SCALING_CACHE`\
    If set, the metric computation will also compute an estimate of the variance of the metric.\
    By default, only compute the metric itself.
* OpenAI configuration
  * `CIK_OPENAI_USE_AZURE`\
    If set to `"True"`, then the baselines using OpenAI models will use the Azure client, instead of the OpenAI client.\
    Default to `"False"`
  * `CIK_OPENAI_API_KEY`\
    Must be set to use baselines using OpenAI models to your API key (either Azure or OpenAI depending on the value of `CIK_OPENAI_USE_AZURE`).
  * `CIK_OPENAI_API_VERSION`\
    If set, select the chosen API version when calling OpenAI model using the Azure client.
  * `CIK_OPENAI_AZURE_ENDPOINT`\
    Select the Azure endpoint to use when calling OpenAI models.
* Nixtla configuration
  * `CIK_NIXTLA_BASE_URL`\
    Must be set to the Nixtla API URL to use the TimeGEN baseline.
  * `NIXTLA_API_KEY`\
    Must be set to your Nixtla API key to use the TimeGEN baseline.

# Data Licenses 

* Dominick's:
[Main webpage](https://www.chicagobooth.edu/research/kilts/research-data/dominicks)
"These data are for academic research purposes only. Users must acknowledge in their working papers and/or publications the Kilts Center for Marketing at the University of Chicago Booth School of Business."
