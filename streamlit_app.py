import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from merged_search import search
from merged_youtube import YouTube
from merged_github import GitHubRepo

# ============================================================
# PAGE CONFIG & INITIALIZATION
# ============================================================
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# SETUP & CONFIGURATION
# ============================================================
load_dotenv()

# API Key setup
key = "AIzaSyBtubnX9o9iPmMMp3L-oq0-t9-uZdcGh6E"
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = key

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# Initialize models & embeddings
@st.cache_resource
def load_models():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embed

llm1, embed = load_models()

# Initialize Git Prompt
git_prompt = PromptTemplate(
    template="""
You are an expert technical assistant.

You will be provided with:
1. A USER QUERY — a question about code, documentation, configuration, or project files.
2. A CONTEXT DICTIONARY — a dictionary where:
   - Each KEY is the name of a vector database (e.g., vectordb_python, vectordb_html, vectordb_java)
   - Each VALUE is the retrieved context text from that specific vector database.

Your responsibilities:
1. Carefully analyze the USER QUERY.
2. From the CONTEXT DICTIONARY, determine which vector database contexts are relevant to answering the query.
   - You may select ONE or MULTIPLE database contexts.
   - Ignore contexts that are not related to the query.
3. Use ONLY the selected relevant contexts to answer the query.
4. Do NOT combine or rely on irrelevant database contexts.
5. Provide:
   - A clear technical explanation.
   - Code snippets or reframed code ONLY if they are present or inferable from the selected context.
6. Maintain strict grounding:
   - Do NOT use external knowledge.
   - Do NOT assume missing information.
   - Do NOT hallucinate APIs, code, or behavior.

If NONE of the provided contexts contain enough information to answer the query, respond EXACTLY with:
"The provided context does not contain enough information to answer this question."

----------------------------------
USER QUERY:
{query}

----------------------------------
CONTEXT DICTIONARY:
{context_dict}

----------------------------------
ANSWER:
""",
    input_variables=["query", "context_dict"]
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🚀 AI Research Assistant")
page = st.sidebar.radio(
    "Select a Tool:",
    ["🔍 Web Search", "📺 YouTube Analysis", "🐙 GitHub Analysis"]
)
gemini_api=st.sidebar.text_input("Google Gemini API Key:" )
tavily_api=st.sidebar.text_input("Tavily API Key:" )
git_access_token=st.sidebar.text_input("GitHub Access Token:" )
# ============================================================
# PAGE: WEB SEARCH
# ============================================================
if page == "🔍 Web Search":
    st.title("🔍 Multi-Source Web Search")
    st.write("Search across YouTube, GitHub, Reddit, and Twitter for relevant information.")
    
    query = st.text_input("Enter your search query:", placeholder="e.g., 'machine learning basics'")
    
    if st.button("🔎 Search", use_container_width=True):
        if query.strip():
            with st.spinner("Searching..."):
                try:
                    s = search(gemini_api=gemini_api, tavily_api=tavily_api)
                    results = s.output_agent_executor(query)
                    
                    # Display final answer
                    if results.get("final_answer"):
                        st.success("Search completed!")
                        st.subheader("📋 AI Summary")
                        st.write(results["final_answer"])
                    
                    # Display GitHub results
                    if results.get("github") and results["github"]:
                        st.subheader("🐙 GitHub Results")
                        for j, item in enumerate(results["github"][0].get("results", [])):
                            with st.expander(f"Result {j + 1}: {item.get('title', 'Untitled')}"):
                                st.write(f"**URL:** {item.get('url')}")
                                st.write(f"**Content:** {item.get('content', 'N/A')}")
                    
                    # Display YouTube results
                    if results.get("youtube") and results["youtube"]:
                        st.subheader("📺 YouTube Results")
                        for j, item in enumerate(results["youtube"][0].get("results", [])):
                            with st.expander(f"Result {j + 1}: {item.get('title', 'Untitled')}"):
                                st.write(f"**URL:** {item.get('url')}")
                    
                    # Display Reddit results
                    if results.get("reddit") and results["reddit"]:
                        st.subheader("🔴 Reddit Results")
                        for j, item in enumerate(results["reddit"][0].get("results", [])):
                            with st.expander(f"Result {j + 1}: {item.get('title', 'Untitled')}"):
                                st.write(f"**URL:** {item.get('url')}")
                    
                    # Display Twitter results
                    if results.get("twitter") and results["twitter"]:
                        st.subheader("𝕏 Twitter/X Results")
                        for j, item in enumerate(results["twitter"][0].get("results", [])):
                            with st.expander(f"Result {j + 1}: {item.get('title', 'Untitled')}"):
                                st.write(f"**URL:** {item.get('url')}")
                    
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        else:
            st.warning("Please enter a search query.")

# ============================================================
# PAGE: YOUTUBE ANALYSIS
# ============================================================
elif page == "📺 YouTube Analysis":
    st.title("📺 YouTube Video Analysis - Interactive")
    st.write("Watch the video and ask continuous questions as you view it.")
    
    # Initialize session state for YouTube
    if "yt_instance" not in st.session_state:
        st.session_state.yt_instance = None
    if "yt_history" not in st.session_state:
        st.session_state.yt_history = []
    if "yt_url" not in st.session_state:
        st.session_state.yt_url = ""
    
    # Video URL input
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        value=st.session_state.yt_url
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        load_video = st.button("▶️ Load & Play Video", use_container_width=True)
    
    with col2:
        clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
    
    # Load video
    if load_video and video_url.strip():
        with st.spinner("Loading video transcript..."):
            try:
                st.session_state.yt_instance = YouTube(embed=embed, llm=llm1)
                st.session_state.yt_instance.extract(video_url)
                st.session_state.yt_url = video_url
                st.session_state.yt_history = []  # Reset chat history
                st.success("✅ Video loaded! Now you can ask questions.")
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
                st.session_state.yt_instance = None
    
    # Clear chat history
    if clear_chat:
        st.session_state.yt_history = []
        st.rerun()
    
    # Display video if loaded
    if st.session_state.yt_instance is not None and st.session_state.yt_url:
        st.divider()
        
        # Extract video ID and display embedded player
        try:
            video_id = st.session_state.yt_url.split("v=")[-1].split("&")[0]
            st.video(f"https://www.youtube.com/embed/{video_id}")
        except:
            st.info("Enter a valid YouTube URL to display the video player.")
        
        st.divider()
        
        # Continuous Q&A section
        st.subheader("💬 Ask Questions About the Video")
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.yt_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
        
        # Question input
        question = st.text_input(
            "Ask a question about the video:",
            placeholder="e.g., 'What are the main topics covered?' or 'Explain the difficulty I'm having...'"
        )
        
        if st.button("❓ Ask Question", use_container_width=True):
            if question.strip():
                with st.spinner("Analyzing..."):
                    try:
                        answer = st.session_state.yt_instance.answer(question)
                        # Add to history
                        st.session_state.yt_history.append((question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error answering question: {str(e)}")
            else:
                st.warning("Please ask a question.")
    
    else:
        if video_url.strip():
            st.info("👆 Click '▶️ Load & Play Video' to start analyzing the video.")
        else:
            st.info("Enter a YouTube URL above to begin.")

# ============================================================
# PAGE: GITHUB ANALYSIS (Interactive Chat)
# ============================================================
elif page == "🐙 GitHub Analysis":
    st.title("🐙 GitHub Repository Analysis - Interactive")
    st.write("Load a repository, then ask continuous questions about the codebase.")

    # Session state
    if "gh_instance" not in st.session_state:
        st.session_state.gh_instance = None
    if "gh_history" not in st.session_state:
        st.session_state.gh_history = []
    if "gh_repo_url" not in st.session_state:
        st.session_state.gh_repo_url = ""

    ACCESS_TOKEN = git_access_token 
    repo_url = st.text_input(
        "Enter GitHub Repository URL:",
        placeholder="https://github.com/username/repo",
        value=st.session_state.gh_repo_url
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        load_repo = st.button("📥 Extract & Load", use_container_width=True)
    with col2:
        clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)

    # Load repository
    if load_repo and repo_url.strip():
        with st.spinner("Extracting repository data..."):
            try:
                gh = GitHubRepo(
                    embed=embed,
                    llm=llm1,
                    prompt=git_prompt,
                    access_token=ACCESS_TOKEN
                )
                gh.extract(repo_url)
                st.session_state.gh_instance = gh
                st.session_state.gh_repo_url = repo_url
                st.session_state.gh_history = []
                st.success("✅ Repository loaded successfully!")
            except Exception as e:
                st.error(f"Error extracting repository: {str(e)}")

    # Clear chat
    if clear_chat:
        st.session_state.gh_history = []
        st.rerun()

    # If repo is loaded, show interactive chat
    if st.session_state.gh_instance is not None and st.session_state.gh_repo_url:
        st.subheader(f"Repository: {st.session_state.gh_repo_url}")
        st.info("Repository loaded. Ask continuous questions below.")

        # Display chat history
        for q, a in st.session_state.gh_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        # Question input
        question = st.text_input(
            "Ask a question about the repository:",
            placeholder="e.g., 'Where is the database connection handled?'"
        )

        if st.button("❓ Ask Question", use_container_width=True):
            if question.strip():
                with st.spinner("Analyzing repository..."):
                    try:
                        answer = st.session_state.gh_instance.answer(question)
                        st.session_state.gh_history.append((question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing repository: {str(e)}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Load a repository first with '📥 Extract & Load'.")

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.info("🤖 Built with LangChain, Streamlit, and Google Gemini AI")
