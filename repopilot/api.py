"""HTTP API. Run with: uvicorn --factory repopilot.api:create_app"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import cache
from typing import Annotated, Any

import psycopg
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi import Path as PathParam
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator

from repopilot import indexer
from repopilot.agent.github import (
    GitHubError,
    Issue,
    IssueNotFound,
    fetch_issue,
    issue_task,
    owner_repo,
)
from repopilot.agent.graph import AgentResult, RepoAgent
from repopilot.config import Settings
from repopilot.embeddings import Embedder, FastEmbedder
from repopilot.ingest import IngestError, normalize_github_url
from repopilot.llm import Answerer, GroqAnswerer
from repopilot.retrieval import FastReranker, Reranker, retrieve
from repopilot.store import Store

log = logging.getLogger(__name__)
IN_PROGRESS = frozenset({"pending", "indexing"})
NO_LLM = "GROQ_API_KEY is not set on the server"

IssueFetcher = Callable[[str, str, int], Issue]


class RepoIn(BaseModel):
    url: str = Field(max_length=300, examples=["https://github.com/pallets/flask"])

    @field_validator("url")
    @classmethod
    def _must_be_github(cls, value: str) -> str:
        try:
            return normalize_github_url(value)
        except IngestError as exc:
            raise ValueError(str(exc)) from exc


class RepoOut(BaseModel):
    id: int
    url: str
    status: str
    error: str | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    path: str
    start_line: int
    end_line: int
    score: float


class AskOut(BaseModel):
    answer: str
    sources: list[Source]


class AgentIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Step(BaseModel):
    tool: str
    args: dict[str, Any]


class AgentOut(BaseModel):
    answer: str
    sources: list[str]  # "path:start-end" for everything the agent's tools returned
    steps: list[Step]


class DraftOut(BaseModel):
    issue_title: str
    issue_url: str
    draft: str  # for a maintainer to review; RepoPilot never posts to GitHub
    sources: list[str]
    steps: list[Step]


def create_app(
    settings: Settings | None = None,
    *,
    store: Store | None = None,
    embedder: Embedder | None = None,
    answerer: Answerer | None = None,
    agent_llm: BaseChatModel | None = None,
    issue_fetcher: IssueFetcher | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = store or Store(settings.database_url, settings.embed_dim)
    if answerer is None and settings.groq_api_key:
        answerer = GroqAnswerer(settings.groq_api_key, settings.llm_model)
    if agent_llm is None and settings.groq_api_key:
        from langchain_groq import ChatGroq

        agent_llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.1,
            timeout=60,
            max_retries=2,
        )
    fetch = issue_fetcher or (
        lambda owner, repo, number: fetch_issue(owner, repo, number, token=settings.github_token)
    )

    @cache
    def get_embedder() -> Embedder:
        # Loaded on first use: the model download should not block startup or health checks.
        return embedder or FastEmbedder(settings.embed_model)

    @cache
    def get_reranker() -> Reranker | None:
        return FastReranker(settings.rerank_model) if settings.rerank else None

    def require_agent() -> RepoAgent:
        if agent_llm is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NO_LLM)
        return RepoAgent(agent_llm, store, get_embedder(), settings.agent_max_tool_rounds)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        store.init_schema()
        yield

    app = FastAPI(title="RepoPilot", version="0.3.0", lifespan=lifespan)

    # Handlers are plain `def`: psycopg calls are blocking, so FastAPI runs them in a threadpool.
    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            store.ping()
        except psycopg.Error as exc:
            log.warning("health check failed: %s", exc)
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "database unavailable"
            ) from exc
        return {"status": "ok"}

    @app.post("/repos", status_code=status.HTTP_202_ACCEPTED, response_model=RepoOut)
    def add_repo(body: RepoIn, background: BackgroundTasks) -> dict:
        repo = store.upsert_repo(body.url)
        if not repo["inserted"] and repo["status"] in IN_PROGRESS:
            return repo  # already queued or running
        store.set_status(repo["id"], "pending")
        # shortcut: in-process background task; move to a job queue for multi-worker deploys.
        background.add_task(
            lambda: indexer.index_repo(store, get_embedder(), settings, repo["id"], body.url)
        )
        return {**repo, "status": "pending", "error": None}

    @app.get("/repos", response_model=list[RepoOut])
    def list_repos() -> list[dict]:
        return store.list_repos()

    @app.get("/repos/{repo_id}", response_model=RepoOut)
    def get_repo(repo_id: int) -> dict:
        return _require_repo(store, repo_id)

    @app.post("/repos/{repo_id}/ask", response_model=AskOut)
    def ask(repo_id: int, body: AskIn) -> AskOut:
        _require_ready(store, repo_id)
        if answerer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NO_LLM)
        hits = retrieve(
            store,
            get_embedder(),
            repo_id,
            body.question,
            body.top_k or settings.top_k,
            mode=settings.search_mode,
            reranker=get_reranker(),
        )
        try:
            text = answerer.answer(body.question, hits)
        except Exception as exc:
            log.exception("LLM call failed for repo %s", repo_id)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "the language model failed") from exc
        return AskOut(answer=text, sources=[Source(**h) for h in hits])

    @app.post("/repos/{repo_id}/agent", response_model=AgentOut)
    def ask_agent(repo_id: int, body: AgentIn) -> AgentOut:
        """Like /ask, but the agent searches and reads files over several steps."""
        _require_ready(store, repo_id)
        result = _run_agent(require_agent(), repo_id, body.question)
        return AgentOut(answer=result.answer, sources=list(result.sources), steps=result.steps)

    @app.post("/repos/{repo_id}/issues/{number}/draft-reply", response_model=DraftOut)
    def draft_reply(repo_id: int, number: Annotated[int, PathParam(ge=1)]) -> DraftOut:
        """Draft a reply to a GitHub issue of this repo. The draft is returned, never posted."""
        repo = _require_ready(store, repo_id)
        agent = require_agent()
        owner, name = owner_repo(repo["url"])
        try:
            issue = fetch(owner, name, number)
        except IssueNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "issue not found") from exc
        except GitHubError as exc:
            log.warning("GitHub request failed for %s/%s#%s: %s", owner, name, number, exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "could not read the issue") from exc
        result = _run_agent(agent, repo_id, issue_task(issue))
        return DraftOut(
            issue_title=issue.title,
            issue_url=issue.url,
            draft=result.answer,
            sources=list(result.sources),
            steps=result.steps,
        )

    return app


def _run_agent(agent: RepoAgent, repo_id: int, task: str) -> AgentResult:
    try:
        return agent.run(repo_id, task)
    except Exception as exc:
        log.exception("agent run failed for repo %s", repo_id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "the agent failed") from exc


def _require_repo(store: Store, repo_id: int) -> dict:
    repo = store.get_repo(repo_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "repository not found")
    return repo


def _require_ready(store: Store, repo_id: int) -> dict:
    repo = _require_repo(store, repo_id)
    if repo["status"] != "ready":
        raise HTTPException(status.HTTP_409_CONFLICT, f"repository is {repo['status']}")
    return repo
