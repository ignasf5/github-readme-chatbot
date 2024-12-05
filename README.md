# GitHub README Chatbot

## Overview

This Streamlit application allows users to interact with a GitHub repository's `README.md` file in a chatbot format. By submitting a query, the bot will search through the sections of the `README.md` file and return the most relevant information based on the user's question. The bot uses natural language processing (NLP) techniques, including TF-IDF vectorization and cosine similarity, to determine the best matches for the given query.

## Features

- **Interactive Chatbot**: Ask questions about the repository's `README.md` content.
- **Multiple Repository Support**: Fetch and combine content from multiple GitHub repositories.
- **Natural Language Processing**: The chatbot uses TF-IDF and cosine similarity to find relevant answers from the repository documentation.
- **Image parser**: Content with images will be displayed.

## Installation

1. Clone the repository to your local machine:
  ```bash
  git clone https://github.com/yourusername/github-readme-chatbot.git
  cd github-readme-chatbot
  ```
2. Build image
  ```
  docker build -t github-readme-chatbot .
  ```
3. Run image
  ```
  docker run -p 8501:8501 github-readme-chatbot
  ```
4. Run image with new default repo
  ```
  docker run -p 8501:8501 -e DEFAULT_REPO_URL="https://github.com/another/repository" github-readme-chatbot
  ```
Or run through terminale ```streamlit run ./app.py```
   
## How it Works

**Key Components**

**ReadmeChatbot class**: This class parses the README.md content, processes sections, and uses TF-IDF for matching queries to the relevant sections.

**Cosine Similarity**: The chatbot calculates cosine similarity between the user's query and sections of the README.md file to determine the most relevant responses.
Fetching README Files: The app fetches README.md files from GitHub repositories via the GitHub API.

**Workflow**

**Initialization**: When the app starts, it loads the README.md from a default GitHub repository.

**User Interaction**: Users can enter a GitHub repository URL to load another repository's README.md. The bot combines sections from both repositories.

**Query Processing**: The user submits queries, and the bot returns the most relevant sections of the README.md files.

**Example Usage**
Default Repository: The app starts with a default repository (https://github.com/ignasf5/chatbot). You can modify the default repository URL in the code.

**Adding Another Repository**: Users can add another repository by entering its GitHub URL in the input field. The bot will process and combine both repositories' README.md files.

**Asking Questions**: Type a question related to the repository's documentation in the chat, and the bot will return the top matching sections.

**Example Interaction**

![image](https://github.com/user-attachments/assets/5f9c9e2b-b13d-49d7-a202-1956b6279fbb)

## Technologies Used
**Streamlit**: A Python library used for building the interactive UI.

**scikit-learn**: Used for natural language processing tasks like TF-IDF vectorization and cosine similarity.

**BeautifulSoup**: Used for parsing and processing the HTML content from README.md files.

**Requests**: Used to fetch raw README.md files from GitHub repositories.

## Troubleshooting
Error fetching README: If you encounter issues fetching the README.md from GitHub, ensure the repository URL is correct and the repository has a README.md file in the default branch (typically main or master).

Slow response times: The first time a repository is processed, it may take a moment to load and process the content. Once the vectors are cached, subsequent queries will be faster.

![image](https://github.com/user-attachments/assets/d95f18b0-d015-4013-a0c9-c80cad1baf26)

![image](https://github.com/user-attachments/assets/d37d6d80-073c-48a5-9e55-db6bc76a2b03)

## Content check

```
# Show the raw README.md content
readme_content = fetch_readme_from_github(default_repo_url)
if st.checkbox("Show raw README.md content"):
    st.code(readme_content, language="markdown")

# Extract chatbot instance for parsed sections
combined_chatbot, _, _ = primary_resources
if st.checkbox("Show parsed sections"):
    st.write(combined_chatbot.sections)
```

![image](https://github.com/user-attachments/assets/0d7cf296-53ac-44f5-84c9-7eaa0ebb9918)


## Additionaly 
Implementeded image parser.

![image](https://github.com/user-attachments/assets/501da4af-8bb6-4bed-8e78-c886655a897f)


