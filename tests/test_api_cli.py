import pytest

from veritas.cli import main


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out.lower()
    assert "veritas demo" in out
    assert "page" in out          # citations printed
    assert "refused" in out       # the CEO question is refused


def test_cli_ask(capsys):
    assert main(["ask", "What is the monthly fee under the services agreement?"]) == 0
    out = capsys.readouterr().out
    assert "75,000" in out or "75000" in out


def test_cli_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_api_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from veritas.serving.api import create_app

    client = TestClient(create_app(seed=True))
    assert client.get("/api/health").json()["status"] == "ok"

    r = client.post("/api/chat", json={"query": "What are the primary risk factors?"})
    body = r.json()
    assert body["grounded"] is True
    assert body["citations"]

    refused = client.post("/api/chat", json={"query": "What is the CEO's home address?"}).json()
    assert refused["grounded"] is False

    up = client.post("/api/upload", json={"title": "Memo",
                                          "text": "=== PAGE 1 ===\nProject Zephyr budget is $2.5 million for 2026."})
    assert up["chunks"] >= 1
    ans = client.post("/api/chat", json={"query": "What is the Project Zephyr budget?"}).json()
    assert ans["grounded"] is True and "2.5" in ans["answer"]

    assert "Veritas" in client.get("/").text
