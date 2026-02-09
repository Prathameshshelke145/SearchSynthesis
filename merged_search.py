from tavily import TavilyClient
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent 
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor,create_react_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import InjectedToolArg
import requests 
from datetime import datetime
from langchain_core.documents import Document
from collections import defaultdict
import os
from tavily import TavilyClient
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor


class search:
    def __init__(self, gemini_api: str, tavily_api: str):
        self.gemini_api = gemini_api
        self.tavily_api = tavily_api
    def output_agent_executor(self, input_text: str):
        # -------------------- Tavily Client --------------------
        tavily = TavilyClient(api_key=self.tavily_api)

        # -------------------- Tools --------------------
        @tool
        def youtube_search(query: str) -> dict:
            """Search YouTube videos related to the query."""
            return tavily.search(query + " site:youtube.com", max_results=5)

        @tool
        def github_search(query: str) -> dict:
            """Search GitHub repositories related to the query."""
            return tavily.search(query + " site:github.com", max_results=5)

        @tool
        def reddit_search(query: str) -> dict:
            """Search Reddit discussions related to the query."""
            return tavily.search(query + " site:reddit.com", max_results=5)

        @tool
        def twitter_search(query: str) -> dict:
            """Search Twitter/X posts related to the query."""
            return tavily.search(
                query + " site:twitter.com OR site:x.com", max_results=5
            )

        # -------------------- LLM --------------------
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.gemini_api
        )
        # -------------------- Prompt --------------------
        prompt = hub.pull("hwchase17/react")

        tools = [
            youtube_search,
            github_search,
            reddit_search,
            twitter_search
        ]

        # -------------------- Agent --------------------
        agent = create_react_agent(
            llm=gemini_llm,
            tools=tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        # -------------------- Query --------------------
        query = (
            f"{input_text} search related concepts on youtube, "
            f"github, reddit and twitter and provide links"
        )

        result = agent_executor.invoke({"input": query})

        # -------------------- Tool Output Parsing --------------------
        tool_data = defaultdict(list)

        for action, observation in result.get("intermediate_steps", []):
            tool_data[action.tool].append(observation)

        dicture = {
            "query": input_text,
            "final_answer": result.get("output"),
            "youtube": tool_data.get("youtube_search", []),
            "github": tool_data.get("github_search", []),
            "reddit": tool_data.get("reddit_search", []),
            "twitter": tool_data.get("twitter_search", [])
        }

        # -------------------- Printing --------------------
        print("-" * 200)

        if dicture["github"]:
            print("GitHub")
            for j, i in enumerate(dicture["github"][0].get("results", [])):
                print(f"----------{j + 1}----------")
                print("Title:", i.get("title"))
                print("Url:", i.get("url"))
                print("Content:", i.get("content"))

        print("-" * 200)

        if dicture["youtube"]:
            print("YouTube")
            for j, i in enumerate(dicture["youtube"][0].get("results", [])):
                print(f"----------{j + 1}----------")
                print("Title:", i.get("title"))
                print("Url:", i.get("url"))

        print("-" * 200)

        if dicture["reddit"]:
            print("Reddit")
            for j, i in enumerate(dicture["reddit"][0].get("results", [])):
                print(f"----------{j + 1}----------")
                print("Title:", i.get("title"))
                print("Url:", i.get("url"))

        print("-" * 200)

        if dicture["twitter"]:
            print("Twitter / X")
            for j, i in enumerate(dicture["twitter"][0].get("results", [])):
                print(f"----------{j + 1}----------")
                print("Title:", i.get("title"))
                print("Url:", i.get("url"))

        print("-" * 200)

        # -------------------- Return Results --------------------
        return dicture
