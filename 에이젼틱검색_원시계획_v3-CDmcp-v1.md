# Agent-search-v3-CDmcp-v1 계획서

> 작성일: 2026-02-17
> 기반: Agent_search_v3 복제

---

## 프로젝트 목표

**기존 HTTP 크롤링의 문제점**:
- 포털(네이버/구글/Brave) 검색 시 봇 탐지 → 차단/CAPTCHA
- fallback 연쇄 (axios → playwright → selenium) → 시간 낭비
- 불안정한 검색 결과

**해결책**: Chrome DevTools Protocol (CDP) 기반 검색
- 실제 브라우저 제어 → 사람처럼 행동
- 차단 가능성 대폭 감소
- fallback 불필요 → 시간 단축

---

## 아키텍처

### 기존 v3 구조
```
search_agent.py
    ↓
smartcrawl-mcp (stdio)
    ↓
smartCrawl_server (Express, port 33003)
    ↓
HTTP 크롤링 (axios/playwright/selenium fallback)
    ↓
네이버/구글/Brave
```

### 신규 CDmcp 구조
```
search_agent.py
    ↓
Python MCP Client (신규 구현) ← 핵심
    ↓
chrome-devtools MCP Server (기존, stdio)
    ↓
Chrome (CDP, port 9222)
    ↓
네이버/구글/Brave (실제 브라우저로 접근)
```

---

## 핵심 구현: Python MCP Client

### MCP 통신 방식
- **프로토콜**: JSON-RPC 2.0 over stdio
- **서버 실행**: `npx @anthropic-ai/mcp-server-chrome-devtools`
- **Python에서 subprocess로 서버 실행 → stdin/stdout 통신**

### 필요한 MCP 도구 (27개 중 핵심)
| 도구 | 용도 |
|------|------|
| `list_pages` | 열린 페이지 목록 |
| `new_page` | 새 탭 열기 |
| `navigate_page` | URL 이동 |
| `take_snapshot` | A11Y 트리 (요소 UID) |
| `fill` | 검색어 입력 |
| `click` | 검색 버튼 클릭 |
| `evaluate_script` | JavaScript 실행 (결과 추출) |
| `take_screenshot` | 디버깅용 스크린샷 |

### 포털별 검색 흐름
```
1. new_page() → 빈 탭 생성
2. navigate_page(url="https://search.naver.com/...") → 네이버 검색 이동
   또는 navigate_page(url="https://www.google.com/search?q=...") → 구글
   또는 navigate_page(url="https://search.brave.com/search?q=...") → Brave
3. take_snapshot() → 페이지 구조 파악
4. evaluate_script() → 검색 결과 추출 (title, url, snippet)
5. 결과 반환
```

---

## 구현 단계

### Phase 1: MCP Client 기본 구현
- [ ] `mcp_client.py` 생성
- [ ] subprocess로 chrome-devtools MCP 서버 실행
- [ ] JSON-RPC 통신 (call_tool, list_tools)
- [ ] 기본 도구 테스트 (list_pages, new_page)

### Phase 2: 포털 검색 함수 구현
- [ ] `cdp_search.py` 생성
- [ ] `search_naver(keyword)` - 네이버 뉴스/웹 검색
- [ ] `search_google(keyword)` - 구글 검색
- [ ] `search_brave(keyword)` - Brave 검색
- [ ] `search_all(keyword)` - 3개 포털 병렬 검색

### Phase 3: search_agent.py 통합
- [ ] 기존 smartcrawl 호출 → CDP 검색으로 교체
- [ ] `call_smartcrawl()` → `call_cdp_search()` 변경
- [ ] 기존 fetch_url은 유지 (Jina Reader)
- [ ] E2E 테스트

### Phase 4: 최적화
- [ ] 탭 재사용 (매번 new_page 대신)
- [ ] 병렬 검색 (asyncio + 여러 탭)
- [ ] 에러 핸들링 (CAPTCHA 감지, 재시도)

---

## 파일 구조 (예상)

```
Agent-search-v3-CDmcp-v1/
├── search_agent.py          # 메인 (수정)
├── mcp_client.py            # MCP 클라이언트 (신규)
├── cdp_search.py            # CDP 기반 포털 검색 (신규)
├── start_sglang.sh          # SGLang 서버 시작
├── outputs/                 # 로그
├── venv/                    # Python 환경
├── smartCrawl_server/       # 제거 예정 (CDP로 대체)
├── smartcrawl-mcp/          # 제거 예정 (CDP로 대체)
└── 에이젼틱검색_원시계획_v3-CDmcp-v1.md  # 이 파일
```

---

## 의존성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Chrome (CDP) | 9222 | 브라우저 (필수) |
| SGLang (AgentCPM-Explore) | 30001 | LLM 추론 |
| Jina Reader | 외부 API | URL 본문 fetch (유지) |

### Chrome 실행 (CDP 모드)
```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-profile
```

### chrome-devtools MCP 서버 위치
```
~/JOB_FOLD/유용한기능추가분/chrome-devtools-mcp
```

---

## 예상 성능 비교

| 항목 | 기존 (HTTP) | CDP 방식 |
|------|-------------|----------|
| 차단 가능성 | 높음 | **낮음** |
| fallback 시간 | 10-30초 | **불필요** |
| 안정성 | 불안정 | **안정** |
| 첫 검색 시간 | 5-15초 | 3-5초 (예상) |

---

## 주의사항

1. **Chrome 필수 실행**: CDP 포트 9222 열려있어야 함
2. **MCP 서버 경로**: `~/JOB_FOLD/유용한기능추가분/chrome-devtools-mcp`
3. **기존 smartCrawl 유지**: 본문 크롤링(fetch_url)은 Jina Reader 그대로 사용
4. **포털 검색만 CDP로 교체**: 전체 구조 변경 아님

---

## 진행 기록

### 2026-02-17: 프로젝트 생성
- Agent_search_v3 복제
- 계획서 작성

### SGLang 사용 이유 (Ollama/LM Studio 불가)

**Tool Calling 형식 차이**:
| 백엔드 | Tool Calling | 비고 |
|--------|-------------|------|
| **SGLang + AgentCPM-Explore** | ✅ `<tool_call>` 텍스트 형식 학습됨 | 채택 |
| Ollama | ❌ API 형식 불완전, 도구 미작동 | 불가 |
| LM Studio | ✅ OpenAI function calling (API 레벨) | 형식 다름 |

**핵심**: AgentCPM-Explore 모델은 `<tool_call>{"name":"search"...}</tool_call>` 형식으로 출력하도록 **특별히 훈련됨**. 일반 Qwen3 모델은 이 형식을 모름.

**search_agent.py 파싱 방식**:
```python
pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
matches = re.findall(pattern, text, re.DOTALL)
```

→ SGLang + AgentCPM-Explore 조합이 필수

### 2026-02-17: Phase 1 완료 - MCP Client
- `mcp_client.py` 생성
- `--browserUrl` 옵션으로 기존 Chrome (port 9222) 연결
- 26개 도구 확인 (click, navigate_page, evaluate_script 등)
- JSON-RPC 2.0 over stdio 통신 구현

### 2026-02-17: Phase 2 완료 - 포털 검색
- `cdp_search.py` 생성
- 네이버/구글/Brave 검색 구현
- **테스트 결과**:
  | 포털 | 결과 수 | 상태 |
  |------|---------|------|
  | Naver | 5개 | ✅ 성공 |
  | Google | 4개 | ✅ 성공 (스니펫 포함) |
  | Brave | 5개 | ✅ 성공 |
  | **All** | **14개** | ✅ 성공 |
- 네이버 셀렉터 `#main_pack` (2026년 새 구조 대응)
- smartcrawl 호환 인터페이스 (`search_with_cdp()`)

### 2026-02-17: Phase 3 완료 - search_agent.py 통합
- `call_smartcrawl()` → CDP 검색으로 교체
- `cdp_search.search_with_cdp()` 호출 (asyncio.to_thread)
- **E2E 테스트 성공**:
  - 쿼리: "삼성전자 주가 전망"
  - 총 시간: ~113초
  - 3턴 완료: 검색 → fetch → 답변 생성
  - 최종 답변: 삼성전자 주가 전망 요약 정상 출력

### 현재 상태
- Phase 1: MCP Client ✅
- Phase 2: 포털 검색 ✅
- Phase 3: search_agent.py 통합 ✅
- Phase 4: 최적화 ✅

### Phase 4 완료: 싱글톤 세션 최적화

#### 구현 내용
1. **싱글톤 MCP 클라이언트** (`cdp_search.py`)
   - 한번 연결하면 프로세스 종료까지 세션 유지
   - 차단 회피: 매번 새 브라우저 열지 않음
   - `get_client()` → 공유 인스턴스 반환

2. **순차 검색 채택** (병렬 대신)
   - 이유: MCP 서버가 단일 Chrome 탭 사용
   - 병렬 실행 시 `navigate_page` 충돌 발생
   - 순차 실행이 안정적 (3개 포털 ~32초)

3. **세션 재사용 효과**
   - 1차 검색: ~34초 (초기화 포함)
   - 2차 이후: ~8초 (초기화 생략)

#### 테스트 결과 (2026-02-17)
```
[1차 검색] 삼성전자 주가 (all)
- Naver: 5건 (10.0초)
- Google: 4건 (12.9초)
- Brave: 5건 (8.7초)
- 총: 14건, 33.75초

[2차 검색] 애플 주가 (brave)
- 세션 재사용: 7.59초
```

#### 병렬화 불가 이유
- chrome-devtools MCP는 "현재 선택된 페이지"에서만 작업
- 동시에 여러 탭 제어 불가 (단일 세션 특성)
- 해결하려면 여러 MCP 서버 인스턴스 필요 (오버엔지니어링)

### 최종 아키텍처
```
search_agent.py
    ↓
cdp_search.py (싱글톤 MCP 클라이언트)
    ↓
chrome-devtools MCP Server (stdio)
    ↓
Chrome (CDP, port 9222) ← 세션 유지
    ↓
네이버 → 구글 → Brave (순차)
```

### 성능 비교
| 항목 | 기존 HTTP (v3) | CDP 방식 |
|------|----------------|----------|
| 차단 가능성 | 높음 | **낮음** |
| fallback 시간 | 10-30초 | **없음** |
| 3포털 검색 | 15-40초 (불안정) | **32초 (안정)** |
| 세션 재사용 | 불가 | **가능** |

### search_agent.py 수정 (CDP 적용)

1. **검색 순차 실행** (`execute_tool` 함수)
   - 변경 전: `asyncio.gather(*tasks)` (병렬)
   - 변경 후: `for q in queries: await search_with_timeout(q)` (순차)
   - 이유: CDP 단일 세션에서 병렬 검색 시 `navigate_page` 충돌

2. **MAX_SEARCH_QUERIES 조정**
   - 변경 전: 5
   - 변경 후: 3 (CDP 순차 검색 특성상 시간 단축)

### E2E 테스트 결과 (2026-02-17)

**쿼리**: "테슬라 FSD 최신 업데이트"

| Turn | 작업 | 결과 | 시간 |
|------|------|------|------|
| 1 | 검색 (3쿼리 순차) | 84건 | 113초 |
| 2 | URL fetch (15개 병렬) | 성공 | 16초 |
| 3 | 답변 생성 | Markdown + 출처 | 34초 |
| **총** | | | **~163초** |

### 배치 검증 결과 (5개 쿼리)

| Q | 쿼리 | 상태 | 건수 | 시간 |
|---|------|------|------|------|
| 1 | 아이폰 17 출시 루머 | ✅ ok | 13 | 34.8초 |
| 2 | 테슬라 FSD 최신 업데이트 | ✅ ok | 13 | 33.2초 |
| 3 | 한국 기준금리 전망 | ✅ ok | 13 | 30.6초 |
| 4 | AI 반도체 시장 동향 2026 | ✅ ok | 13 | 30.5초 |
| 5 | 서울 아파트 전세시장 전망 | ✅ ok | 12 | 33.0초 |

- **성공률**: 5/5 (100%)
- **총 검색 결과**: 64건
- **세션 재사용**: ✅ 확인 (MCP 초기화 1회만)

### 2026-02-18: 캐시 기능 전체 제거

#### 제거된 기능
| 기능 | 설명 | 제거 이유 |
|------|------|----------|
| **로컬 인덱스** | SQLite FTS5 (`local_index.db`) | 항상 최신 검색 필요 |
| **검색 캐시** | JSON (`outputs/cache/search/`) | 캐시된 결과 대신 실시간 검색 |
| **Fetch 캐시** | JSON (`outputs/cache/fetch/`) | 항상 최신 콘텐츠 fetch |
| **도메인 힌트** | JSON (`domain_hints.json`) | 캐시와 함께 불필요 |

#### 유지된 기능
- **Chrome 세션 싱글톤**: MCP 클라이언트 1회 초기화 후 세션 유지
- **봇 감지 회피**: 실제 브라우저 세션으로 차단 방지

#### 제거된 코드
```python
# 제거된 상수
SEARCH_CACHE_TTL, FETCH_CACHE_TTL, MAX_CACHE_CONTENT
CACHE_DIR, SEARCH_CACHE_DIR, FETCH_CACHE_DIR, DOMAIN_HINTS_PATH

# 제거된 함수
_hash_key(), _cache_get(), _cache_set()
_load_domain_hints(), _save_domain_hints()
_get_domain_hint(), _set_domain_hint()
_fetch_cache_path(), _get_fetch_cache(), _set_fetch_cache()
_init_local_index(), _index_document(), _search_local_index()

# 제거된 import
sqlite3, hashlib
```

#### 성능 변화
| 항목 | 캐시 O | 캐시 X |
|------|--------|--------|
| 3쿼리 검색 | ~0.1초 (캐시 히트) | **~110초** (CDP 실제 검색) |
| URL fetch | ~0.1초 (캐시 히트) | **~10-20초** (실제 fetch) |
| 전체 E2E | ~60초 | **~150-180초** |

**결론**: 속도는 느려졌지만 **항상 최신 정보** 검색 보장

### 2026-02-18: 탭 3개 재사용 최적화

#### 기존 문제
- 매 검색마다 탭 계속 생성 → **탭 폭증** (스크린샷 참고)
- MCP가 탭 목록을 제대로 못 봄 (CDP 직접 조회로 해결)

#### 새 아키텍처
```
[초기화 - 1회만]
1. CDP API로 기존 탭 모두 정리
2. 포털별 탭 3개 생성 (naver, google, brave)
3. 탭 인덱스 매핑 (0, 1, 2)

[검색 시 - 탭 재사용]
select_page(탭) → navigate_page(검색URL) → evaluate_script(추출)
```

#### 유틸리티
- `clear_tabs.py`: 수동 탭 정리 (CDP 직접)
- `clear_tabs_cdp.sh`: 쉘 스크립트 버전

#### 성능 개선
| 항목 | v1 (순차 navigate) | v2 (탭 재사용) |
|------|-------------------|----------------|
| 3포털 검색 | ~40초 | **~20초** |
| 탭 관리 | 탭 폭증 | **3개 고정** |

#### 주의사항
- 크롬 CDP 포트 필수: `google-chrome --remote-debugging-port=9222`
- 탭 많으면 `python clear_tabs.py`로 정리
- 크롬 종료되면 다시 시작 필요

### 2026-02-18: 크롬 3개 인스턴스 분리 (진짜 병렬)

#### 기존 문제
- 탭 3개 방식에서도 MCP가 마지막 탭만 재사용
- 단일 크롬 + 탭 전환 = 순차 실행 (병렬 아님)

#### 새 아키텍처 - 크롬 3개 독립 실행
```
[Chrome 1] port 9222 - Naver 전용
[Chrome 2] port 9223 - Google 전용
[Chrome 3] port 9224 - Brave 전용
          ↓
    ThreadPoolExecutor (진짜 병렬)
          ↓
    CDP 직접 통신 (MCP 없이)
```

#### 신규 파일
| 파일 | 역할 |
|------|------|
| `chrome_launcher.py` | 크롬 3개 인스턴스 관리 (start/stop/status) |
| `cdp_search.py` | CDP 직접 통신 + 병렬 검색 (완전 재작성) |

#### chrome_launcher.py 핵심
```python
CHROME_INSTANCES = {
    "naver": {"port": 9222, "profile": "/tmp/chrome-naver-profile"},
    "google": {"port": 9223, "profile": "/tmp/chrome-google-profile"},
    "brave": {"port": 9224, "profile": "/tmp/chrome-brave-profile"}
}

# 사용법
python chrome_launcher.py start   # 3개 모두 시작
python chrome_launcher.py stop    # 3개 모두 종료
python chrome_launcher.py status  # 상태 확인
python chrome_launcher.py restart # 재시작
```

#### cdp_search.py 핵심
- MCP 제거 → CDP WebSocket 직접 통신
- `ThreadPoolExecutor(max_workers=3)`로 진짜 병렬
- 각 크롬에 독립 세션 유지 (`user-data-dir`)

#### 성능 비교
| 항목 | 탭 3개 (순차) | 크롬 3개 (병렬) |
|------|--------------|----------------|
| 3포털 검색 | ~33초 | **~5초** |
| 병렬 실행 | ❌ | ✅ |
| 세션 분리 | ❌ (공유) | ✅ (독립) |

### 2026-02-18: 탐지 회피 + 캡챠 감지

#### 탐지 회피 플래그 추가 (`chrome_launcher.py`)
```python
cmd = [
    "google-chrome",
    "--remote-debugging-port={port}",
    "--user-data-dir={profile}",
    "--remote-allow-origins=*",
    # 탐지 회피 플래그
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--disable-infobars",
    "--disable-extensions",
    "--disable-popup-blocking",
    "--ignore-certificate-errors",
    ...
]
```

#### 캡챠 감지 로직 (`cdp_search.py`)
```python
CAPTCHA_DETECT_SCRIPT = """
    var indicators = [];
    // hCaptcha, Cloudflare Turnstile, reCAPTCHA 감지
    // 텍스트 기반: "verify you are human" 등
    // URL 기반: "challenge", "captcha" 포함
    return indicators;
"""

# 결과 0건이면 자동으로 캡챠 체크
if len(results) == 0:
    captcha_indicators = client.evaluate(CAPTCHA_DETECT_SCRIPT)
    if captcha_indicators:
        print("🚨 [BRAVE] 캡챠 감지! 브라우저에서 수동 해결 필요")
```

#### 캡챠 감지 시 출력 예시
```
🚨 [BRAVE] 캡챠 감지! 브라우저(port 9224)에서 캡챠를 풀어주세요.
```

### 2026-02-18: 타임아웃 최적화

#### 변경된 설정 (`search_agent.py`)
| 설정 | 변경 전 | 변경 후 | 이유 |
|------|---------|---------|------|
| `MAX_PARALLEL` | 15 | **50** | 병렬 fetch 증가 |
| `FETCH_TIMEOUT` | 20초 | **3초** | 느린 URL 빨리 포기 |
| `FETCH_LOCAL_TIMEOUT` | 10초 | **2초** | 로컬 추출 타임아웃 |
| `FETCH_GLOBAL_BUDGET` | 45초 | **10초** | 전체 fetch 예산 단축 |

#### E2E 테스트 결과 (최적화 후)
```
검색 (3쿼리 × 3포털 병렬): ~13초
Fetch 공식 URL 5개: ~1.5초
Fetch 28개 URL (병렬): ~6초 (이전 35초)
모델 응답: ~60초
총: ~80초
```

### 현재 아키텍처 (v3.1)
```
search_agent.py
    ↓
cdp_search.py (CDP 직접 통신)
    ↓
┌─────────────────────────────────────┐
│  Chrome 1    Chrome 2    Chrome 3  │
│  port 9222   port 9223   port 9224 │
│  Naver       Google      Brave     │
│  (세션유지)   (세션유지)   (세션유지)  │
└─────────────────────────────────────┘
    ↓ ThreadPoolExecutor (병렬)
    ↓
검색 결과 병합
```

### 파일 구조 (최종)
```
Agent-search-v3-CDmcp-v1/
├── search_agent.py          # 메인 에이전트
├── cdp_search.py            # CDP 직접 검색 (병렬)
├── chrome_launcher.py       # 크롬 3개 관리
├── start_sglang.sh          # SGLang 서버 시작
├── outputs/                 # 로그
├── venv/                    # Python 환경
└── 에이젼틱검색_원시계획_v3-CDmcp-v1.md
```

### 사용법
```bash
# 1. 크롬 3개 시작
cd ~/JOB_FOLD/LLM_test/Agent-search-v3-CDmcp-v1
source venv/bin/activate
python chrome_launcher.py start

# 2. 검색 테스트
python -c "from cdp_search import search_with_cdp; print(search_with_cdp('테스트'))"

# 3. E2E 실행
python search_agent.py "검색 질문"
python search_agent.py "검색 질문" --depth simple   # 빠른 검색
python search_agent.py "검색 질문" --depth medium   # 기본값
python search_agent.py "검색 질문" --depth deep     # 심층 검색

# 4. 크롬 재시작 (캡챠 문제 시)
python chrome_launcher.py restart
```

---

## Phase 5: 검색 깊이 3단계 (2026-02-18)

### 배경
- Turn 3에서 146초 소요 (타임아웃 직전)
- 원인: 28개 URL fetch 결과 = 109KB → ~80,000 토큰
- LLM 속도는 입력 토큰에 비례 (O(n²) attention)

### 해결: 검색 깊이 옵션

| 레벨 | 방식 | 입력 토큰 | 예상 시간 |
|------|------|----------|----------|
| **simple** | snippet만 | ~5K | ~10초 |
| **medium** | snippet + fetch 5개 | ~25K | ~30초 |
| **deep** | snippet + fetch 전체 | ~80K | ~150초 |

### 각 레벨 용도
- **simple**: 빠른 개요, 단순 질문 ("삼성전자 주가?")
- **medium**: 일반 질문 ("삼성전자 2026년 전망?") - 기본값
- **deep**: 심층 리서치 ("HBM4 전략 경쟁사 비교 분석")

### 구현
- `--depth` 인자 추가
- simple: fetch_url 비활성화
- medium: MAX_FETCH_URLS = 5
- deep: 현재 방식 유지

### 테스트 결과 (2026-02-18)

**컨텍스트 크기 비교**:
| 모드 | 도구결과 크기 | 추정 토큰 |
|------|-------------|----------|
| simple | 4,508자 | ~3K |
| medium | 24,915자 | ~17K |
| deep | 115,267자 | ~77K |

**실측 시간** (삼성전자 2026년 실적 전망 쿼리):
| 모드 | Turn 3 모델 | 총 시간 |
|------|------------|---------|
| simple | ~6초 | ~35초 |
| medium | ~12초 | ~50초 |
| deep | ~146초 | ~170초 |

**결론**: medium 모드가 최적 (deep 대비 3배 빠르면서 충분한 정보)

### Claude WebSearch 비교 (2026-02-18)

동일 쿼리 "삼성전자 2026년 실적 전망"으로 비교:

| 방식 | 시간 | 토큰 | 정보량 | 형식 |
|------|------|------|--------|------|
| **Claude WebSearch** | 44초 | N/A | 중간 | 표/정돈됨 |
| Agent simple | 35초 | ~3K | 적음 | bullet |
| Agent medium | 50초 | ~17K | 중간 | bullet |
| Agent deep | 170초 | ~77K | 많음 | 상세 |

**WebSearch 장점**:
- Anthropic 서버에서 검색 → 봇 차단 없음
- 결과가 표/마크다운으로 정돈됨
- 별도 인프라 불필요

**Agent 장점**:
- 로컬 실행 (오프라인 가능)
- 깊이 조절 가능 (simple/medium/deep)
- 커스텀 검색 로직 (포털 선택, 필터링)

**용도별 추천**:
- 빠른 답변: simple (35초)
- 균형: WebSearch (44초) 또는 medium (50초)
- 심층 분석: deep (170초)
