# LLM-Agent for E-Commerce Recommendation

This project leverages an LLM agent framework with user-comment-simulation and actor-critic architecture for personalized recommendations in e-commerce tasks. It supports both the MovieLens and Amazon datasets.

---

## Datasets
- **MovieLens**: [Dataset Link](https://grouplens.org/datasets/movielens/)  
- **Amazon**: [Dataset Link](https://amazon-reviews-2023.github.io/index.html)  

Pre-processed datasets are included in the repository. You can directly run the scripts without re-downloading or preprocessing the data.

---

## Quick Start
- We use the AutoGen multi-agent framework and GPT-4o Api for our project. Make sure you have OPENAI_API_KEY and export it to your environment.
- Ensure all packages are installed by running.

To run the project, execute the following scripts based on your desired task:

- **For MovieLens Task**:
  ```bash
  python movie_demo/code/test.py
  ```

- **For Amazon Task**:
  ```bash
  python amazon_test.py
  ```
