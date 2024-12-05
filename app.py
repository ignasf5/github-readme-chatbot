import streamlit as st
import logging
import requests
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from markdown import markdown
from bs4 import BeautifulSoup

# Get the logging level from the environment variable (default to 'INFO')
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()

# Configure logging
logging.basicConfig(
    level=log_level,  # Set log level from environment variable
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Get the repository URL from the environment variable
default_repo_url = os.getenv("DEFAULT_REPO_URL", "https://github.com/ignasf5/chatbot")  # Default fallback if not set

# Get the max summary length from the environment variable (default to 500 if not set)
max_summary_length = int(os.getenv("SUMMARY_MAX_LENGTH", "500"))

# Get the values from environment variables
page_title = os.getenv("PAGE_TITLE", "Chatbot")
title = os.getenv("TITLE", "GitHub README Chatbot")

st.set_page_config(
    page_title=page_title,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title(title)

st.markdown(
            r"""
        <style>
        .stAppDeployButton {
                visibility: hidden;
            }
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        </style>
        """, unsafe_allow_html=True
    )

# Initialize chat history and state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "additional_repo_message" not in st.session_state:
    st.session_state.additional_repo_message = ""

# Class for the chatbot that processes README content
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from markdown import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Class for the chatbot that processes README content
class ReadmeChatbot:
    """
    A chatbot that processes a README.md file, parses it into sections, 
    and enables searching of relevant sections based on a user query.

    Attributes:
        content (str): The raw content of the README file.
        sections (dict): A dictionary where keys are section titles and values are section content.
        cleaned_titles (list): A list of cleaned section titles for easier matching.
    """

    def __init__(self, content):
        """
        Initializes the ReadmeChatbot with the content of the README.md file.

        Args:
            content (str): The raw content of the README file.
        
        Initializes the following:
            - content: Stores the README content.
            - sections: A dictionary of parsed sections from the README.
            - cleaned_titles: A list of cleaned section titles.
        """
        logger.debug(f"Initializing ReadmeChatbot with content length: {len(content)}")
        self.content = content
        self.sections = self.parse_readme()  # Parse the README into sections
        self.cleaned_titles = [self.clean_text(title) for title in self.sections.keys()]  # Clean the section titles
        logger.debug(f"Parsed README into sections: {list(self.sections.keys())}")
        logger.debug(f"Cleaned section titles: {self.cleaned_titles}")

    def parse_readme(self):
        """
        Parses the README.md content into sections by identifying headers (h1, h2, h3)
        and grouping subsequent content under these headers.

        Returns:
            dict: A dictionary where keys are section titles (headers) and values are the content of the sections.
        
        Example:
            {
                "Introduction": "This is the intro section content.",
                "Installation": "Steps to install the software...",
                ...
            }
        """
        logger.debug("Parsing README content into sections...")
        soup = BeautifulSoup(markdown(self.content), 'html.parser')
        sections = {}
        current_section = "Introduction"  # Default section if no header is found
        sections[current_section] = []  # Start the first section

        # Iterate through the parsed HTML content to find headers and content
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'img']):
            if element.name in ['h1', 'h2', 'h3']:  # New section header found
                current_section = element.text.strip()  # Update the current section
                sections[current_section] = []  # Start a new section
            elif element.name == 'img':  # Image found
                img_url = element['src']
                sections[current_section].append(f"![Image]({img_url})")  # Markdown for image
            else:
                sections[current_section].append(element.text.strip())  # Add text content

        logger.debug(f"Parsed README into {len(sections)} sections.")
        return {k: ' '.join(v) for k, v in sections.items()}  # Combine paragraphs and images in each section

    def search_query(self, query, vectorizer, section_vectors, threshold=0.1):
        """
        Searches for the most relevant sections based on a user query using cosine similarity.

        Args:
            query (str): The query string input by the user.
            vectorizer (TfidfVectorizer): The TF-IDF vectorizer used to transform the text data.
            section_vectors (array): The precomputed vectors of the sections of the README.
            threshold (float, optional): The minimum cosine similarity score for a section to be considered relevant. Default is 0.1.
        
        Returns:
            list: A list of tuples containing the section title, content, and similarity score for relevant matches.
        """
        logger.debug(f"Searching for query: {query}")
        cleaned_query = self.clean_text(query)  # Clean the query
        logger.info(f"Cleaned query: {cleaned_query}")

        query_vector = vectorizer.transform([cleaned_query])  # Transform the query into a vector
        logger.debug(f"Vector for query: {query_vector.toarray()}")

        cosine_sim = cosine_similarity(query_vector, section_vectors).flatten()  # Calculate cosine similarity
        logger.debug(f"Cosine similarity scores: {cosine_sim}")

        # Find the relevant sections that have a similarity score above the threshold
        keys = list(self.sections.keys())
        corpus = list(self.sections.values())
        relevant_matches = [
            (keys[i], corpus[i], cosine_sim[i]) 
            for i in range(len(cosine_sim)) if cosine_sim[i] > threshold
        ]

        # Sort matches by the cosine similarity score (descending order)
        relevant_matches.sort(key=lambda x: x[2], reverse=True)
        logger.debug(f"Found {len(relevant_matches)} relevant matches.")
        return relevant_matches

    def clean_text(self, text):
        """
        Cleans the input text by converting it to lowercase and removing special characters.
        
        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text, ready for processing.
        """
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text))  # Remove non-alphanumeric characters
        return text

# Fetch README file content from a GitHub repository
def fetch_readme_from_github(repo_url):
    """Fetch the README.md file from a GitHub repository."""
    try:
        logger.info(f"Fetching README from URL: {repo_url}")
        if "raw.githubusercontent.com" in repo_url:
            response = requests.get(repo_url)
        else:
            parts = repo_url.rstrip("/").split("/")
            owner, repo = parts[-2], parts[-1]
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
            response = requests.get(raw_url)
        
        response.raise_for_status()
        logger.info(f"Successfully fetched README content from {repo_url}.")
        return response.text
    except Exception as e:
        st.error(f"Failed to fetch README.md: {e}")
        logger.error(f"Failed to fetch README.md from {repo_url}: {e}")
        return None

# Precompute the TF-IDF vectors from the README content
@st.cache_resource
def precompute_tfidf_vectors(content):
    """
    Precompute TF-IDF vectors for README content.

    This function processes the README content by:
    1. Initializing a ReadmeChatbot to parse and clean the content.
    2. Combining section titles and their respective content into a corpus.
    3. Using TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize the corpus.
    4. Returning the chatbot instance, vectorizer, and the computed vectors.

    Args:
        content (str): The content of the README file.

    Returns:
        tuple: (chatbot, vectorizer, section_vectors)
            - chatbot (ReadmeChatbot): Parsed and cleaned README content.
            - vectorizer (TfidfVectorizer): TF-IDF vectorizer used for the transformation.
            - section_vectors (sparse matrix): Vectorized representation of README sections.

    Notes:
        - The result is cached using `st.cache_resource` to improve performance on repeated calls.
    """
    logger.debug("Precomputing TF-IDF vectors for README content...")

    # Parse README content using ReadmeChatbot
    chatbot = ReadmeChatbot(content)
    
    # Combine section titles and content into a single corpus
    all_texts = [f"{title} {content}" for title, content in chatbot.sections.items()]
    
    # Vectorize the corpus using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=None)
    section_vectors = vectorizer.fit_transform(all_texts)

    logger.debug("Vectorized sections with combined titles and content successfully.")
    logger.debug(f"TF-IDF feature names: {vectorizer.get_feature_names_out()}")
    
    return chatbot, vectorizer, section_vectors


# Fetch and process a repository
def fetch_and_process_repository(repo_url):
    """Fetch and process the README content from a GitHub repository."""
    logger.info(f"Processing repository: {repo_url}")
    readme_content = fetch_readme_from_github(repo_url)
    if readme_content:
        return precompute_tfidf_vectors(readme_content)
    else:
        logger.error(f"Failed to process repository: {repo_url}")
        return None

# Combine resources from two repositories
def combine_resources(primary, additional):
    """Combine resources from two repositories."""
    if not additional:
        logger.warning("No additional resources to combine.")
        return primary
    
    logger.debug("Combining primary and additional resources...")
    combined_chatbot = primary[0]
    combined_chatbot.sections.update(additional[0].sections)

    combined_texts = [
        f"{title} {content}" for title, content in combined_chatbot.sections.items()
    ]
    vectorizer = TfidfVectorizer(stop_words=None)
    combined_section_vectors = vectorizer.fit_transform(combined_texts)
    
    logger.debug("Combined resources successfully.")
    return combined_chatbot, vectorizer, combined_section_vectors

# Generate a response from the chatbot
def generate_bot_response(prompt, chatbot, vectorizer, section_vectors, threshold=0.1):
    results = chatbot.search_query(prompt, vectorizer, section_vectors, threshold=threshold)
    if results:
        response = ""
        for i, (title, text, score) in enumerate(results[:3]):
            response += f"### {i + 1}. {title} (Score: {score:.2f})\n\n{text[:max_summary_length]}...\n\n"
            image_urls = re.findall(r'!\[Image]\((.*?)\)', text)
            for img_url in image_urls:
                response += f"![Image]({img_url})\n\n"
    else:
        response = "Sorry, I couldn't find any relevant information."
    return response


# Load the default repository
primary_resources = fetch_and_process_repository(default_repo_url)
if not primary_resources:
    logger.error(f"Failed to load primary resources for {default_repo_url}")
    st.error("Failed to load the default repository.")
    st.stop()

# Show the raw README.md content
readme_content = fetch_readme_from_github(default_repo_url)
if st.checkbox("Show raw README.md content"):
    st.code(readme_content, language="markdown")

# Extract chatbot instance for parsed sections
combined_chatbot, _, _ = primary_resources
if st.checkbox("Show parsed sections"):
    st.write(combined_chatbot.sections)

st.session_state.primary_resources = primary_resources

# Allow the user to add another repository
# st.markdown("### Add Another Repository")

additional_repo_url = st.text_input("Enter the GitHub repository URL:")

if additional_repo_url:
    additional_resources = fetch_and_process_repository(additional_repo_url)
    if additional_resources:
        st.session_state.additional_resources = additional_resources
        st.session_state.primary_resources = combine_resources(
            st.session_state.primary_resources,
            additional_resources
        )
        st.session_state.additional_repo_message = f"Repository added successfully! {additional_repo_url}"
    else:
        st.session_state.additional_repo_message = "Failed to add repository. Please check the URL."

# Display the message after adding a repository
if st.session_state.additional_repo_message:
    st.success(st.session_state.additional_repo_message)
    st.session_state.additional_repo_message = ""

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Message"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate bot response
    combined_chatbot, vectorizer, section_vectors = st.session_state.primary_resources
    response = generate_bot_response(prompt, combined_chatbot, vectorizer, section_vectors, threshold=0.1)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
