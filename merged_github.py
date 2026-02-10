import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
import cssutils
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import GithubFileLoader

class GitHubRepo:
    def __init__(self, embed, llm, prompt, git_type : str, access_token: str):
        self.embed = embed
        self.llm = llm
        self.prompt = prompt
        self.git_type = git_type
        self.access_token = access_token
        self.vectordbs = {}
    def html_to_code_chunks(self, html_text):
        soup = BeautifulSoup(html_text, "html.parser")
        chunks = []

        BLOCK_TAGS = [
            "section", "article", "main", "div",
            "form", "table", "ul", "ol",
            "header", "footer", "nav"
        ]

        for tag in soup.find_all(BLOCK_TAGS, recursive=True):
            html_block = str(tag)
            if len(html_block) > 80:
                chunks.append(html_block)

        for tag in soup.find_all(["canvas", "script", "style"]):
            chunks.append(str(tag))

        return chunks
    def chunk_css(self, css_text):
        sheet = cssutils.parseString(css_text)
        chunks = []

        for rule in sheet:
            if rule.type == rule.STYLE_RULE:
                chunks.append(
                    Document(
                        page_content=f"{rule.selectorText} {{ {rule.style.cssText} }}",
                        metadata={"type": "css"}
                    )
                )
        return chunks

    def extract_java_code(self, java_data):
        cleaned = []
        for code in java_data:
            code = re.sub(r"/\*[\s\S]*?\*/", "", code)
            code = re.sub(r"//.*", "", code)
            code = "\n".join(line for line in code.splitlines() if line.strip())
            cleaned.append(code)
        return cleaned

    def chunk_java(self, clean_code):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA,
            chunk_size=2000,
            chunk_overlap=150
        )

        chunks = []
        for code in clean_code:
            chunks.extend(splitter.split_text(code))
        return chunks
    
    def _simple_language(self, docs, ext, lang, name, size):
        text = "\n\n".join(d.page_content for d in docs if d.metadata["path"].endswith(ext))
        if text:
            splitter = RecursiveCharacterTextSplitter.from_language(lang, size, 0)
            self.vectordbs[name] = Chroma.from_texts(
                splitter.split_text(text), self.embed, collection_name=name
            )

    def _simple_text(self, docs, exts, name):
        text = "\n\n".join(d.page_content for d in docs if d.metadata["path"].endswith(exts))
        if text:
            splitter = RecursiveCharacterTextSplitter(2000, 150)
            self.vectordbs[name] = Chroma.from_texts(
                splitter.split_text(text), self.embed, collection_name=name
            )

    def _html_loader(self, docs):
        html = "".join(d.page_content for d in docs if d.metadata["path"].endswith(".html"))
        if html:
            self.vectordbs["html"] = Chroma.from_texts(
                self.html_to_code_chunks(html), self.embed, "html_db"
            )

    def _css_loader(self, docs):
        css = "".join(d.page_content for d in docs if d.metadata["path"].endswith(".css"))
        if css:
            self.vectordbs["css"] = Chroma.from_documents(
                self.chunk_css(css), self.embed, "css_db"
            )

    def _java_loader(self, docs):
        java = [d.page_content for d in docs if d.metadata["path"].endswith(".java")]
        if java:
            chunks = self.chunk_java(self.extract_java_code(java))
            self.vectordbs["java"] = Chroma.from_texts(
                chunks, self.embed, "java_db"
            )

    def extract(self, link: str):
        # Parse repo name from URL
        n = link.find("com") + 4
        repo_name = link[n:]
        print(f"Extracting from repo: {repo_name}")

        ACCESS_TOKEN = self.access_token
        loader = GithubFileLoader(
            repo=repo_name,
            branch=self.git_type,
            access_token=ACCESS_TOKEN,
            github_api_url="https://api.github.com",
            file_filter=lambda f: f.endswith((
                ".py",
                ".js", ".jsx", ".ts", ".tsx",
                ".html", ".css", ".scss",
                ".json", ".yaml", ".yml", ".toml", ".ini",
                ".md", ".txt", ".rst",
                ".c", ".h", ".cpp", ".hpp",
                ".java", ".kt",
                ".go", ".rs",
                ".sh", ".bash",
                ".sql",
                ".ipynb"
            ))
        )

        documents = loader.load()

        # Build vector stores per file type and collect into a map
        vectordb_map = {}

        # IPYNB
        ipynb_data = [doc.page_content for doc in documents if doc.metadata["path"].endswith(".ipynb")]
        if ipynb_data:
            ipdata = json.loads(ipynb_data[0])
            cell_texts = [
                "".join(cell["source"]).strip()
                for cell in ipdata.get("cells", [])
                if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
            ]
            if cell_texts:
                vectordb_map["ipynb"] = Chroma.from_texts(cell_texts, self.embed, collection_name="ipynb_notebooks")

        # PYTHON
        python_code = "\n\n".join(doc.page_content for doc in documents if doc.metadata["path"].endswith(".py"))
        if python_code:
            splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=800, chunk_overlap=100)
            python_chunks = splitter.split_text(python_code)
            vectordb_map["python"] = Chroma.from_texts(python_chunks, self.embed, collection_name="python_notebooks")

        # JAVASCRIPT
        js_data = "\n\n".join(doc.page_content for doc in documents if doc.metadata["path"].endswith(".js"))
        if js_data:
            splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=2000, chunk_overlap=0)
            chunks_js = splitter.split_text(js_data)
            vectordb_map["js"] = Chroma.from_texts(chunks_js, self.embed, collection_name="js_notebooks")

        # HTML
        html_text = "".join(doc.page_content for doc in documents if doc.metadata["path"].endswith(".html"))
        if html_text:
            html_chunks = self.html_to_code_chunks(html_text)
            vectordb_map["html"] = Chroma.from_texts(html_chunks, self.embed, collection_name="html_notebooks")

        # CSS
        css_text = "".join(doc.page_content for doc in documents if doc.metadata["path"].endswith(".css"))
        if css_text:
            chunks_css = self.chunk_css(css_text)
            vectordb_map["css"] = Chroma.from_documents(chunks_css, self.embed, collection_name="css_notebooks")

        # JAVA
        java_data = [doc.page_content for doc in documents if doc.metadata["path"].endswith(".java")]
        if java_data:
            clean_code = self.extract_java_code(java_data)
            if clean_code:
                chunks_java = self.chunk_java(clean_code)
                vectordb_map["java"] = Chroma.from_texts(chunks_java, self.embed, collection_name="java_notebooks")

        # JSON
        json_data = [doc.page_content for doc in documents if doc.metadata["path"].endswith(".json")]
        if json_data:
            vectordb_map["json"] = Chroma.from_texts(json_data, self.embed, collection_name="json_notebooks")

        # YML / YAML
        yml_data = [doc.page_content for doc in documents if doc.metadata["path"].endswith((".yml", ".yaml"))]
        if yml_data:
            vectordb_map["yml"] = Chroma.from_texts(yml_data, self.embed, collection_name="yml_notebooks")

        # TEXT / MD
        text_data = [doc.page_content for doc in documents if doc.metadata["path"].endswith((".md", ".txt"))]
        if text_data:
            text_text = "".join(text_data)
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
            chunks = splitter.split_text(text_text)
            vectordb_map["text"] = Chroma.from_texts(chunks, self.embed, collection_name="text_notebooks")

        # Save to self.vectordbs for later querying
        self.vectordbs = vectordb_map
        return vectordb_map
       
    def answer(self,git_hub_key:str,query: str):
        # if self.llm is None:
        #     raise ValueError("LLM is not initialized. Check API key or model loading.")
        githun_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=git_hub_key
        )
        context_dict = {}

        for name, db in self.vectordbs.items():
            docs = db.similarity_search(query)
            context_dict[name] = [d.page_content for d in docs]

        final_prompt = self.prompt.format(
            query=query,
            context_dict=context_dict
        )

        return githun_llm.invoke(final_prompt).content
       