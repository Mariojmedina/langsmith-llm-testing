# LLM Testing and Evaluation with LangSmith

Large‑language models (LLMs) can behave unpredictably in production.  Even a simple change in wording can trigger hallucinations or expose latency problems.  **LangSmith** is a developer‑focused platform designed to make building and testing LLM applications easier and more reliable.  It provides a unified suite of tools for debugging, evaluation, performance monitoring and observability【379625664723094†L49-L53】.  Instead of juggling multiple dashboards and logging tools, LangSmith gives you real‑time visibility into how your model behaves — tracking prompts, model responses, tool calls and memory usage in one place【379625664723094†L62-L71】.  This centralised view helps you trace errors quickly and fix them before they affect users【379625664723094†L92-L103】.

This repository demonstrates how to set up a small, reproducible evaluation pipeline using LangSmith.  It contains a Python script and a Jupyter notebook that show how to create a dataset of questions and expected answers, write a simple chain (or model function) to produce responses, and measure its performance.  The examples intentionally use a dummy chain so you can run them without API keys or proprietary models; however, the same workflow works with any LangChain chain or LLM you choose.

## Why LangSmith?

* **Unified observability and debugging.**  LangSmith traces the flow of data through your application so you can identify bottlenecks and misaligned prompts.  Its dashboards let you see each step of your model’s reasoning process【379625664723094†L181-L200】.
* **Evaluation framework.**  The platform includes built‑in evaluators for metrics such as accuracy and relevance, and supports both automated tests and manual reviews【379625664723094†L181-L200】.  This allows you to benchmark models and compare alternative chains.
* **Real‑time monitoring.**  LangSmith continuously monitors key performance indicators such as latency and error rates.  When something goes wrong it alerts you immediately, helping teams move from reactive firefighting to proactive optimisation【379625664723094†L92-L103】【379625664723094†L204-L207】.
* **Integration with LangChain.**  LangSmith plugs into the LangChain ecosystem so you can evaluate chains built with multiple models and tools【379625664723094†L181-L200】.  It also supports open‑source tools, making it accessible to a broad range of developers【379625664723094†L209-L213】.

## Repository structure

```text
langsmith-llm-testing/
├── README.md              # This overview and setup guide
├── LICENSE                # MIT license for this project
├── requirements.txt       # Python dependencies
├── scripts/
│   └── test_llm_langsmith.py  # Command‑line example for evaluating a dummy chain
└── notebooks/
    └── langsmith_llm_testing.ipynb  # Interactive demo of dataset creation and evaluation
```

## Getting started

1. **Install dependencies.**  Create a virtual environment and install the requirements:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create a LangSmith account (optional).**  To log evaluations to the LangSmith cloud, sign up at `https://smith.langchain.com` and generate an API key.  Set `LANGSMITH_API_KEY` and optionally `LANGSMITH_PROJECT` as environment variables.  If you do not set these variables the examples will run locally without uploading results.

3. **Run the script.**  Execute the command‑line example:

   ```bash
   python scripts/test_llm_langsmith.py --dataset demo
   ```

   The script loads a small dataset of questions and expected answers, runs a dummy chain that returns answers, and prints a simple accuracy report.  If you have configured your API key, the dataset and evaluation results will also be uploaded to your LangSmith dashboard.

4. **Explore the notebook.**  Launch the Jupyter notebook to walk through the evaluation process interactively:

   ```bash
   jupyter notebook notebooks/langsmith_llm_testing.ipynb
   ```

   The notebook explains how to create a dataset programmatically, define a chain, run evaluations and inspect results with pandas.  Because it uses a dummy chain, it will work out of the box without external API calls.

## Extending this example

To test a real LLM:

1. Replace the dummy chain in `scripts/test_llm_langsmith.py` and the notebook with your own LangChain chain or call to an LLM (e.g., an `OpenAI` model, an `ollama` model, or a custom agent).  See the LangChain documentation for examples on building chains.
2. Add more examples to the dataset.  The more diverse your evaluation set, the more reliable your metrics will be.
3. Use built‑in evaluators.  LangSmith provides many out‑of‑the‑box evaluators (such as `qa_exact_match`, `embedding_distance` and `code_eval`) as well as the ability to write custom evaluators.
4. Run evaluations regularly.  Continuous evaluation helps track regressions and improvements over time.  Real‑time monitoring and systematic performance evaluation are key to reliable AI applications【379625664723094†L230-L302】.

## Limitations

While LangSmith is powerful, it does have some limitations:

* **Learning curve.**  New users must understand prompting, model evaluation and DevOps workflows before LangSmith makes sense【379625664723094†L133-L144】.
* **Ecosystem dependence.**  LangSmith is tightly integrated with LangChain; using it outside of the LangChain ecosystem may require additional effort【379625664723094†L146-L152】.
* **Cost and scalability.**  Large‑scale projects may incur significant costs for evaluation runs and trace storage【379625664723094†L160-L165】.  Always monitor your usage when scaling up.

## Importance of statistical evaluation

When evaluating LLM systems you are effectively measuring the reliability of a safety‑critical component.  A misbehaving agent could delete a database record or expose sensitive information, so rigorous testing isn’t just academic — it’s the difference between a robust assistant and a costly mistake【988116894173750†L55-L58】【988116894173750†L135-L139】.  Statistical evaluation helps you quantify performance and catch issues early.  Model evaluation is a core task in the machine learning workflow that improves predictive power and ensures models work correctly in production【988116894173750†L60-L63】.

Common metrics include **accuracy**, **precision**, **recall**, **error rate**, **confusion matrix** and **F1‑score**【988116894173750†L74-L77】.  Accuracy measures the proportion of correct predictions【988116894173750†L94-L102】 but can be misleading on imbalanced datasets; precision and recall provide complementary views by focusing on false positives and false negatives.  Evaluating a model with multiple metrics helps you understand its strengths and weaknesses【988116894173750†L74-L77】.  Cross‑validation and hold‑out splits help estimate generalisation, and evaluation can reveal overfitting when a model performs well on training data but poorly on new data【988116894173750†L119-L123】.

Throughout this project, the notebook demonstrates how to compute these metrics and interpret them.  Although the dummy chain used here cannot delete a database, the same statistical rigour applies to real agents handling critical tasks.

## References

This README summarises information from Nitor Infotech’s article *“LangSmith: The Key to Reliable LLM Applications”*.  The article explains that LangSmith offers tools for debugging, evaluation, performance monitoring and observability【379625664723094†L49-L53】, provides real‑time visibility into model behaviour【379625664723094†L62-L71】, consolidates debugging, testing and monitoring into a single dashboard【379625664723094†L92-L103】 and includes an evaluation framework with metrics like accuracy and relevance【379625664723094†L181-L200】.