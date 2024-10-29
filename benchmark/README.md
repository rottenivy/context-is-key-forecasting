# Context is Key: A Benchmark for Forecasting with Essential Textual Information

📝 [[Paper]](https://arxiv.org/abs/2410.18959) 🌐 [[Website]](https://servicenow.github.io/context-is-key-forecasting) ✉️ [[Contact]](mailto:arjun.ashok@servicenow.com,andrew.williams1@servicenow.com,alexandre.drouin@servicenow.com)

![banner](https://github.com/user-attachments/assets/ca15c5f6-a1cc-4db0-891b-767dad4f241e)




## Setting environment variables

Here is a list of all environment variables which the Context-is-Key benchmark will access:

| Variable Name               | Description                                                                                     | Default Value                        |
|-----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------|
| **CIK_MODEL_STORE**         | Folder to store model weights for the baselines.                                                | `./models`                           |
| **CIK_DATA_STORE**          | Folder to store downloaded datasets.                                                            | `./data`                     |
| **CIK_DOMINICK_STORE**      | Folder to store the Dominick dataset for specific tasks.                                        | `CIK_DATA_STORE + /dominicks`        |
| **CIK_TRAFFIC_DATA_STORE**  | Folder to store the Traffic dataset for specific tasks.                                         | `CIK_DATA_STORE + /traffic_data`     |
| **HF_HOME**                 | Cache location for downloading datasets from Hugging Face.                                      | `CIK_DATA_STORE + /hf_cache`         |
| **CIK_RESULT_CACHE**        | Folder to store the output of baselines to avoid recomputation.                                 | `./inference_cache`                  |
| **CIK_METRIC_SCALING_CACHE**| Folder to store scaling factors for each task to avoid recomputation.                           | `./metric_scaling_cache`             |
| **CIK_METRIC_COMPUTE_VARIANCE** | If set, computes an estimate of the variance of the metric.                              | Only compute metric itself by default|
| **CIK_OPENAI_USE_AZURE**    | If set to "True", use Azure client instead of OpenAI client for baselines using OpenAI models.  | `False`                              |
| **CIK_OPENAI_API_KEY**      | API key for accessing OpenAI models.                                    | None (Required for baseline)         |
| **CIK_OPENAI_API_VERSION**  | API version for OpenAI models when using the Azure client.                                     | None                                 |
| **CIK_OPENAI_AZURE_ENDPOINT** | Azure endpoint for calling OpenAI models.                                                    | None                                 |
| **CIK_LLAMA31_405B_URL**    | API URL for the Llama-3.1-405b baseline.                                | None (Required for baseline)         |
| **CIK_LLAMA31_405B_API_KEY**| API key for the Llama-3.1-405b API.                                     | None (Required for baseline)         |
| **CIK_NIXTLA_BASE_URL**     | Azure API URL for the Nixtla TimeGEN baseline.                             | None (Required for baseline)         |
| **CIK_NIXTLA_API_KEY**          | Azure API key for the Nixtla TimeGEN baseline.                          | None (Required for baseline)         |
