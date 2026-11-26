from repopilot.ingest import Chunk
from repopilot.store import Store
from tests.conftest import HashEmbedder, requires_db

pytestmark = requires_db

AUTH = Chunk("src/auth.py", 1, 3, "def login(user, password):\n    return verify_password(user)")
CACHE = Chunk("src/cache.py", 1, 2, "class LRUCache:\n    evicts least recently used entries")
DOCS = Chunk("README.md", 1, 2, "# Setup\nInstall dependencies with pip")


def _index(store: Store, embedder: HashEmbedder, url: str, chunks: list[Chunk]) -> int:
    repo = store.upsert_repo(url)
    vectors = embedder.embed_documents([f"{c.path}\n{c.content}" for c in chunks])
    store.replace_chunks(repo["id"], chunks, vectors)
    return repo["id"]


def test_upsert_repo_is_idempotent(store: Store) -> None:
    first = store.upsert_repo("https://github.com/a/b")
    second = store.upsert_repo("https://github.com/a/b")

    assert first["inserted"] is True
    assert second["inserted"] is False
    assert first["id"] == second["id"]
    assert first["status"] == "pending"
    assert [r["url"] for r in store.list_repos()] == ["https://github.com/a/b"]


def test_set_status_records_error_and_chunk_count(store: Store) -> None:
    repo_id = store.upsert_repo("https://github.com/a/b")["id"]

    store.set_status(repo_id, "failed", error="boom")
    failed = store.get_repo(repo_id)
    store.set_status(repo_id, "ready", chunk_count=7)
    ready = store.get_repo(repo_id)

    assert (failed["status"], failed["error"]) == ("failed", "boom")
    assert (ready["status"], ready["error"], ready["chunk_count"]) == ("ready", None, 7)
    assert store.get_repo(999) is None


def test_search_ranks_the_relevant_chunk_first(store: Store, embedder: HashEmbedder) -> None:
    repo_id = _index(store, embedder, "https://github.com/a/b", [AUTH, CACHE, DOCS])
    question = "how does login verify the password"

    hits = store.search(repo_id, question, embedder.embed_query(question), k=3)

    assert hits[0]["path"] == "src/auth.py"
    assert (hits[0]["start_line"], hits[0]["end_line"]) == (1, 3)
    assert hits[0]["score"] > hits[-1]["score"]


def test_search_finds_exact_identifier_by_keyword(store: Store, embedder: HashEmbedder) -> None:
    repo_id = _index(store, embedder, "https://github.com/a/b", [AUTH, CACHE, DOCS])

    # Unrelated query vector, so only the keyword half of hybrid search can find LRUCache.
    hits = store.search(repo_id, "LRUCache", embedder.embed_query("pip install"), k=1)

    assert hits[0]["path"] == "src/cache.py"


def test_search_is_scoped_to_one_repo(store: Store, embedder: HashEmbedder) -> None:
    repo_a = _index(store, embedder, "https://github.com/a/a", [AUTH])
    _index(store, embedder, "https://github.com/b/b", [CACHE, DOCS])

    hits = store.search(repo_a, "cache", embedder.embed_query("cache"), k=10)

    assert [h["path"] for h in hits] == ["src/auth.py"]


def test_search_modes_use_only_their_own_signal(store: Store, embedder: HashEmbedder) -> None:
    repo_id = _index(store, embedder, "https://github.com/a/b", [AUTH, CACHE, DOCS])
    unrelated_vector = embedder.embed_query("pip install dependencies setup")

    keyword = store.search(repo_id, "LRUCache", unrelated_vector, k=3, mode="keyword")
    vector = store.search(repo_id, "LRUCache", unrelated_vector, k=1, mode="vector")

    assert [h["path"] for h in keyword] == ["src/cache.py"]  # keyword mode returns matches only
    assert vector[0]["path"] == "README.md"  # vector mode follows the (unrelated) vector


def test_replace_chunks_drops_the_previous_index(store: Store, embedder: HashEmbedder) -> None:
    repo_id = _index(store, embedder, "https://github.com/a/b", [AUTH, CACHE])
    store.replace_chunks(repo_id, [DOCS], embedder.embed_documents([DOCS.content]))

    hits = store.search(repo_id, "login", embedder.embed_query("login"), k=10)

    assert [h["path"] for h in hits] == ["README.md"]
