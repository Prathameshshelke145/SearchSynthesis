from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


class YouTube():
    def __init__(self, embed, llm) :
        self.embed = embed
        self.llm = llm
        self.vectordb_store = None

    def extract(self, link: str):
        loader = YoutubeLoader.from_youtube_url(
            link,
            add_video_info=False
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No transcript found for this video.")

        self.vectordb_store = Chroma.from_texts(
            texts=[doc.page_content for doc in chunks],
            embedding=self.embed,
            collection_name="youtube_notebook"
        )

        return "Video processed successfully"

    def answer(self,youtube_api_key: str, query: str):
        if self.vectordb_store is None:
            raise ValueError("Please extract a YouTube video first.")
        youtube_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=youtube_api_key
        )
        retriever = self.vectordb_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "lambda_mult": 1}
        )
       
        docs = retriever.invoke(query)
        context = " ".join(doc.page_content for doc in docs)

        prompt = PromptTemplate(
            template=(
                "You are an assistant. Answer the question: {question} "
                "based on the following context:\n{context}"
            ),
            input_variables=["question", "context"]
        )

        final_prompt = prompt.format(
            question=query,
            context=context
        )

        response = youtube_llm.invoke(final_prompt)
        return response.content.replace("\n", " ")
