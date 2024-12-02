import streamlit as st
import logging
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from markdown import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

st.title("GitHub README Chatbot")

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
class ReadmeChatbot:
    def __init__(self, content):
        logger.debug(f"Initializing ReadmeChatbot with content length: {len(content)}")
        self.content = content
        self.sections = self.parse_readme()
        self.cleaned_titles = [self.clean_text(title) for title in self.sections.keys()]
        logger.debug(f"Parsed README into sections: {list(self.sections.keys())}")
        logger.debug(f"Cleaned section titles: {self.cleaned_titles}")

    def parse_readme(self):
        """Parse the README.md content into sections."""
        logger.debug("Parsing README content into sections...")
        soup = BeautifulSoup(markdown(self.content), 'html.parser')
        sections = {}
        current_section = "Introduction"
        sections[current_section] = []

        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            if element.name in ['h1', 'h2', 'h3']:
                current_section = element.text.strip()
                sections[current_section] = []
            else:
                sections[current_section].append(element.text.strip())

        logger.debug(f"Parsed README into {len(sections)} sections.")
        return {k: ' '.join(v) for k, v in sections.items()}

    def search_query(self, query, vectorizer, section_vectors, threshold=0.1):
        """Search for the most relevant sections based on the query."""
        logger.debug(f"Searching for query: {query}")
        cleaned_query = self.clean_text(query)
        logger.info(f"Cleaned query: {cleaned_query}")

        query_vector = vectorizer.transform([cleaned_query])
        logger.debug(f"Vector for query: {query_vector.toarray()}")

        cosine_sim = cosine_similarity(query_vector, section_vectors).flatten()
        logger.debug(f"Cosine similarity scores: {cosine_sim}")

        keys = list(self.sections.keys())
        corpus = list(self.sections.values())
        relevant_matches = [
            (keys[i], corpus[i], cosine_sim[i]) 
            for i in range(len(cosine_sim)) if cosine_sim[i] > threshold
        ]

        relevant_matches.sort(key=lambda x: x[2], reverse=True)
        logger.debug(f"Found {len(relevant_matches)} relevant matches.")
        return relevant_matches

    def clean_text(self, text):
        """Function to clean input text by converting to lowercase and removing special characters."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text))
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
    """Precompute TF-IDF vectors for the README content."""
    logger.debug("Precomputing TF-IDF vectors for README content...")
    chatbot = ReadmeChatbot(content)
    all_texts = [f"{title} {content}" for title, content in chatbot.sections.items()]
    vectorizer = TfidfVectorizer(stop_words=None)
    section_vectors = vectorizer.fit_transform(all_texts)
    logger.debug("Vectorized sections with combined titles and content successfully")
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
    """Generate a response based on the user's query."""
    logger.debug(f"Generating bot response for prompt: {prompt}")
    results = chatbot.search_query(prompt, vectorizer, section_vectors, threshold=threshold)
    if results:
        response = "Here are the top matches:\n\n"
        for i, (title, text, score) in enumerate(results[:3]):  # Limit to top 3 results
            response += f"**{i + 1}. {title}** (Score: {score:.2f})\n{text[:300]}...\n\n"
    else:
        response = "Sorry, I couldn't find any relevant information."
    logger.debug(f"Generated response: {response[:100]}...")  # Log first 100 characters of the response
    return response

# Load the default repository
default_repo_url = "https://github.com/ignasf5/chatbot"
primary_resources = fetch_and_process_repository(default_repo_url)
if not primary_resources:
    logger.error(f"Failed to load primary resources for {default_repo_url}")
    st.stop()

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
