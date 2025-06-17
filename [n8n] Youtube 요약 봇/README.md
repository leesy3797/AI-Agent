---
## 🚀 n8n YouTube 구독 채널 새 동영상 알림 및 요약 텔레그램 워크플로우 기획안

이 워크플로우는 YouTube 구독 채널에 새 동영상이 업로드되면 텔레그램으로 알림을 보내고, 사용자가 텔레그램에서 "요약 시작" 버튼을 클릭하면 해당 동영상을 요약하여 다시 텔레그램으로 전송하는 것을 목표로 합니다.

### 🎯 목표
* YouTube 구독 채널의 새 동영상 업로드 실시간 감지 🔍
* 텔레그램을 통한 즉각적인 알림 전송 🔔
* 사용자 요청에 따른 YouTube 동영상 요약 및 텔레그램 전송 자동화 🤖

---

### 🧩 주요 구성 요소 (n8n 노드 기준)

* **Trigger (트리거):**
    * **YouTube Trigger (Watch New Videos in Channel)**
        * **설정:**
            * **Resource:** `Channel`
            * **Channel ID(s):** 알림을 받고자 하는 YouTube 채널 ID (쉼표로 구분하여 여러 개 추가 가능)
            * **Events:** `New Video`
            * **Poll Interval:** 5분 (적절히 조절 가능)
        * **역할:** 지정된 YouTube 채널에 새로운 동영상이 업로드될 때마다 워크플로우를 시작합니다. ▶️

* **Initial Notification (초기 알림):**
    * **Telegram 노드 (Send Message)**
        * **설정:**
            * **Chat ID:** 사용자 텔레그램 Chat ID (개인 또는 그룹)
            * **Text:**
                ```
                새 동영상이 업로드되었습니다! 🎬
                채널: {{ $json.channelName }}
                제목: {{ $json.title }}
                링크: {{ $json.url }}

                동영상 요약을 원하시면 아래 버튼을 클릭해주세요.
                ```
            * **Reply Markup:** `Inline Keyboard`
                * **Button 1:**
                    * **Text:** `🎥 요약 시작`
                    * **Callback Data:** `summarize_video:{{ $json.videoId }}` (동영상 ID를 포함하여 고유하게 식별)
        * **역할:** 새 동영상 정보를 사용자에게 텔레그램으로 알리고, 요약 요청을 위한 인라인 버튼을 제공합니다. 💬

* **Telegram Callback Query Handler (텔레그램 콜백 쿼리 핸들러):**
    * **Telegram Trigger (On Callback Query)**
        * **설정:**
            * **Chat ID:** (비워두면 모든 챗에 반응)
        * **역할:** 텔레그램 메시지의 인라인 버튼 클릭 (콜백 쿼리)을 감지합니다. 👆

* **Condition (조건문):**
    * **If 노드**
        * **설정:**
            * **Value 1:** `{{ $json.callback_query.data.startsWith("summarize_video:") }}`
            * **Operation:** `Is True`
        * **역할:** 콜백 데이터가 "summarize_video:"로 시작하는지 확인하여 요약 요청인지 판별합니다. ✅

* **Extract Video ID (동영상 ID 추출):**
    * **Code 노드 (JavaScript)**
        * **설정:**
            ```javascript
            const callbackData = $json.callback_query.data;
            const videoId = callbackData.split(":")[1];
            return [{ json: { videoId: videoId } }];
            ```
        * **역할:** 콜백 데이터에서 동영상 ID를 추출하여 다음 노드로 전달합니다. ✂️

* **Get Video Transcript/Caption (동영상 스크립트/캡션 가져오기):**
    * **YouTube 노드 (Get Video Caption)**
        * **설정:**
            * **Video ID:** `{{ $json.videoId }}` (이전 Code 노드에서 추출한 값)
            * **Language:** `ko` (한국어 자막 우선, 없으면 영어 등 fallback 로직 추가 고려)
        * **역할:** 지정된 YouTube 동영상의 자막 또는 스크립트를 가져옵니다. (자막이 없는 경우 요약 불가) 📝

* **Summarize Text (텍스트 요약):**
    * **OpenAI 노드 (GPT-3/4 또는 기타 LLM API)**
        * **설정:**
            * **Model:** `gpt-3.5-turbo` (또는 `gpt-4` 등)
            * **Prompt:**
                ```
                다음 YouTube 동영상 스크립트를 한국어로 300자 내외로 요약해주세요.

                스크립트:
                {{ $json.data.caption }}
                ```
        * **역할:** 가져온 동영상 스크립트를 LLM API를 사용하여 요약합니다. 🧠✨

* **Send Summary to Telegram (요약 텔레그램 전송):**
    * **Telegram 노드 (Send Message)**
        * **설정:**
            * **Chat ID:** `{{ $json.callback_query.message.chat.id }}` (콜백을 보낸 사용자에게 응답)
            * **Text:**
                ```
                [동영상 요약]

                {{ $json.choices[0].message.content }}

                원본 링크: {{ $json.url }}
                ```
        * **역할:** 요약된 내용을 사용자 텔레그램으로 전송합니다. 📤

* **Error Handling (에러 핸들링 - 선택 사항이지만 권장):**
    * **Try/Catch 블록**
        * 요약 실패 (자막 없음, LLM API 오류 등) 시 사용자에게 실패 메시지를 보냅니다.
    * **Telegram 노드 (Send Message)**
        * **설정:**
            * **Chat ID:** `{{ $json.callback_query.message.chat.id }}`
            * **Text:** `동영상 요약 중 오류가 발생했습니다. 잠시 후 다시 시도해 주시거나, 동영상에 자막이 없는 경우 요약이 어려울 수 있습니다. 😅`
        * **역할:** 요약 과정에서 문제가 발생했을 때 사용자에게 알립니다. ⚠️

---

### 🗺️ 워크플로우 로직 흐름 (개념도)

```mermaid
graph TD
    A[YouTube Trigger: 새 동영상 감지] --> B{Telegram: 초기 알림 전송 + 요약 버튼};
    B --> C{Telegram Trigger: 콜백 쿼리 감지};
    C -- "callback_query.data == 'summarize_video:...'" --> D{If: 요약 요청인가?};
    D -- "True" --> E[Code: 동영상 ID 추출];
    E --> F[YouTube: 동영상 자막 가져오기];
    F --> G[OpenAI: 스크립트 요약];
    G --> H[Telegram: 요약 결과 전송];
    D -- "False" --> I[End: 다른 콜백 처리 (미구현)];
    F -- "자막 없음 / 오류" --> J[Error Handling: 텔레그램 오류 메시지 전송];
    G -- "요약 오류" --> J;
```

---

### 📋 세부 기획 고려 사항

* **API Key 관리:** YouTube Data API Key, Telegram Bot Token, OpenAI API Key는 n8n Credentials에 **안전하게 저장**하고 사용해야 합니다. 🔑
* **YouTube 채널 ID 확보:** 알림을 받고자 하는 YouTube 채널의 ID를 미리 파악해야 합니다. 🆔
* **텔레그램 Chat ID 확보:** 개인 텔레그램 Chat ID를 미리 파악해야 합니다. (봇과의 대화를 통해 `getUpdates` API 등으로 확인 가능) 🗣️
* **자막 없는 동영상 처리:** YouTube `Get Video Caption` 노드에서 자막을 가져오지 못할 경우에 대한 **예외 처리**를 명확히 해야 합니다. (예: "자막이 없어 요약이 어렵습니다" 메시지 전송) 🚫字幕
* **LLM 요약 품질:** OpenAI 프롬프트를 통해 요약 품질을 개선할 수 있습니다. 300자 내외 등 길이를 명시하거나, 특정 요약 스타일을 지시할 수 있습니다. 🌟
* **비용 고려:** OpenAI API 사용 시 토큰 사용량에 따라 비용이 발생합니다. 💰
* **워크플로우 활성화/비활성화:** 필요에 따라 워크플로우를 쉽게 켜고 끌 수 있도록 합니다. 🔛
* **에러 로깅:** n8n의 기본 에러 로깅 기능을 활용하여 문제 발생 시 디버깅에 활용합니다. 🐞
* **보안:** 텔레그램 콜백 데이터에 민감 정보가 포함되지 않도록 주의합니다. 🔒
* **여러 채널 처리:** `YouTube Trigger`에서 여러 채널 ID를 설정하여 한 워크플로우로 여러 채널을 모니터링할 수 있습니다. 📡
* **요약 요청 버튼의 유효성:** 초기 알림 메시지에서 보낸 동영상에 대한 요약 버튼만 유효하도록 `callback_data`에 `videoId`를 포함하는 것이 중요합니다. ✅
* **텔레그램 메시지 편집:** 요약이 완료된 후 초기 알림 메시지를 편집하여 요약이 완료되었음을 표시하거나, 해당 버튼을 비활성화하는 것도 좋은 사용자 경험이 될 수 있습니다 (Telegram `Edit Message` 노드 활용). ✏️