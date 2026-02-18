# LocalWebSearch-CDP

**API 키 없는 로컬 WebSearch**

로컬 LLM (Ollama, LM Studio)에서 WebSearch 기능이 필요하신가요?
API 키 발급 없이, **Chrome만 있으면 됩니다.**

## 핵심 가치

| 항목 | Tavily/Brave API | LocalWebSearch-CDP |
|------|------------------|---------------------|
| API 키 | **필요** | **불필요** |
| 비용 | 유료/제한 | **무료** |
| 설치 난이도 | 복잡 | **Chrome만 설치** |
| 한국 포털 | 제한적 | **네이버 지원** |

## 특징

- **API 키 불필요** - Chrome CDP (DevTools Protocol) 직접 제어
- **병렬 검색** - 크롬 3개 인스턴스로 네이버/구글/Brave 동시 검색
- **한국어 지원** - 네이버 통합검색 포함
- **로컬 LLM 연동** - SGLang + AgentCPM-Explore 기반
- **봇 탐지 회피** - 세션 유지 + 탐지 회피 플래그

## 설치

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Chrome 시작 (디버깅 모드)
```bash
# 크롬 3개 인스턴스 자동 시작
python chrome_launcher.py
```

### 3. SGLang 서버 시작 (LLM)
```bash
./start_sglang.sh
```

## 사용법

```bash
# 단일 질문
python search_agent.py "검색어"

# 검색 깊이 설정
python search_agent.py "검색어" --depth simple   # snippet만 (빠름)
python search_agent.py "검색어" --depth medium   # 상위 5개 URL fetch
python search_agent.py "검색어" --depth deep     # 전체 URL fetch (느림)

# 대화형 모드
python search_agent.py -i
```

## 아키텍처

```
사용자 질문
    ↓
SGLang (port 30001) - AgentCPM-Explore 4B
    ↓
CDP Search (병렬)
├── Chrome:9222 → 네이버
├── Chrome:9223 → 구글
└── Chrome:9224 → Brave
    ↓
최종 답변 + 출처
```

## 파일 구조

```
LocalWebSearch-CDP/
├── search_agent.py      # 메인 에이전트
├── cdp_search.py        # CDP 병렬 검색
├── chrome_launcher.py   # 크롬 인스턴스 관리
├── start_sglang.sh      # SGLang 서버 시작
├── clear_tabs.py        # 탭 정리 유틸리티
└── clear_tabs_cdp.sh    # 탭 정리 스크립트
```

## 성능

| 모드 | 시간 | 토큰 수 |
|------|------|---------|
| simple | ~35초 | ~3K |
| medium | ~50초 | ~17K |
| deep | ~170초 | ~77K |

## 요구사항

- Python 3.10+
- Chrome/Chromium
- SGLang + AgentCPM-Explore 모델

## 라이선스

MIT License

## 기여

포털 추가 PR 환영합니다! `cdp_search.py`의 `PORTAL_CONFIG` 주석을 참고하세요.

---

Co-Authored-By: inchul <insung8150@users.noreply.github.com>
Co-Authored-By: Claude <noreply@anthropic.com>
