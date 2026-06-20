"""Codeforces public API client: rate-limited, backed off, paginated (docs/06)."""
import time

import httpx
from pydantic import BaseModel, Field

BASE_URL = "https://codeforces.com/api/"
RETRY_STATUS = {429, 503}


class CFError(RuntimeError):
    pass


class CFProblem(BaseModel):
    contest_id: int | None = Field(default=None, alias="contestId")
    index: str = ""
    name: str = ""
    rating: int | None = None
    tags: list[str] = []

    @property
    def pid(self) -> str:
        return f"{self.contest_id}{self.index}"


class CFSubmission(BaseModel):
    id: int
    creation_time: int = Field(alias="creationTimeSeconds")
    problem: CFProblem
    verdict: str | None = None
    author: dict = {}

    @property
    def participant_type(self) -> str:
        return self.author.get("participantType", "")


class CFRatingChange(BaseModel):
    contest_id: int = Field(alias="contestId")
    new_rating: int = Field(alias="newRating")
    update_time: int = Field(alias="ratingUpdateTimeSeconds")


class CFUserInfo(BaseModel):
    handle: str
    rating: int | None = None
    max_rating: int | None = Field(default=None, alias="maxRating")
    rank: str | None = None


class CFClient:
    def __init__(self, transport=None, min_interval: float = 1.5, max_retries: int = 4,
                 page_size: int = 1000) -> None:
        self._client = httpx.Client(base_url=BASE_URL, transport=transport, timeout=30.0)
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._page_size = page_size
        self._last_request = 0.0

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, method: str, **params):
        backoff = 2.0
        for attempt in range(self._max_retries):
            self._throttle()
            resp = self._client.get(method, params=params)
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if resp.status_code in RETRY_STATUS:
                if attempt < self._max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            if resp.status_code != 200 or body.get("status") != "OK":
                raise CFError(f"{method}: HTTP {resp.status_code} {body.get('comment', resp.text[:200])}")
            return body["result"]
        raise CFError(f"{method}: exhausted retries")

    def problemset_problems(self) -> tuple[list[CFProblem], dict[str, int]]:
        result = self._get("problemset.problems")
        problems = [CFProblem.model_validate(p) for p in result.get("problems", [])]
        solved: dict[str, int] = {}
        for stat in result.get("problemStatistics", []):
            pid = f"{stat.get('contestId')}{stat.get('index')}"
            solved[pid] = stat.get("solvedCount", 0)
        return problems, solved

    def user_status(self, handle: str) -> list[CFSubmission]:
        out: list[CFSubmission] = []
        frm = 1
        while True:
            page = self._get("user.status", handle=handle, **{"from": frm, "count": self._page_size})
            out.extend(CFSubmission.model_validate(s) for s in page)
            if len(page) < self._page_size:
                break
            frm += self._page_size
        return out

    def user_rating(self, handle: str) -> list[CFRatingChange]:
        return [CFRatingChange.model_validate(r) for r in self._get("user.rating", handle=handle)]

    def user_info(self, handle: str) -> CFUserInfo:
        result = self._get("user.info", handles=handle)
        return CFUserInfo.model_validate(result[0])
