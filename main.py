import json

import requests

BASE_URL = "http://localhost:11434/api"
MODEL_NAME = "hf.co/unsloth/Qwen3-4B-GGUF:q2_K_L"

HEADERS = {
    "Content-Type": "application/json"
}


class OllamaAPIClient:
    def __init__(self, base_url=BASE_URL, model_name=MODEL_NAME):
        self.base_url = base_url
        self.model_name = model_name
        self.headers = HEADERS

    def list_models(self):
        """설치된 모델 목록을 가져옵니다."""
        try:
            response = requests.get(f"{self.base_url}/tags", headers=self.headers)
            response.raise_for_status()

            models_data = response.json()
            if "models" in models_data:
                return models_data["models"]
            else:
                print("❌ 모델 정보를 찾을 수 없습니다.")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ 모델 목록 조회 실패: {e}")
            return []

    def generate_completion(self, prompt, stream=False, **kwargs):
        """단일 프롬프트로 텍스트 생성 (completion)"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        try:
            if stream:
                return self.proc_stream_response("/generate", data)
            else:
                return self.proc_response("/generate", data, "response")

        except requests.exceptions.RequestException as e:
            print(f"❌ Generate API 호출 실패: {e}")
            return None

    def chat_completion(self, messages, stream=False, **kwargs):
        """대화 형식으로 텍스트 생성 (chat)"""
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        try:
            if stream:
                return self.proc_stream_response("/chat", data)
            else:
                return self.proc_response("/chat", data, "message")

        except requests.exceptions.RequestException as e:
            print(f"❌ Chat API 호출 실패: {e}")
            return None

    def proc_stream_response(self, endpoint, data):
        """스트리밍 응답을 처리합니다."""
        full_response = ""

        with requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                data=json.dumps(data),
                stream=True
        ) as response:
            response.raise_for_status()

            print("🔄 스트림 응답:")
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    # completion endpoint의 경우
                    if "response" in chunk:
                        content = chunk["response"]
                        full_response += content
                        print(content, end="", flush=True)
                    # chat endpoint의 경우
                    elif "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_response += content
                        print(content, end="", flush=True)

        print("\n✅ 스트림 완료!")
        return full_response

    def proc_response(self, endpoint, data, response_key):
        """일반 응답을 처리합니다."""
        response = requests.post(
            f"{self.base_url}{endpoint}",
            headers=self.headers,
            data=json.dumps(data)
        )
        response.raise_for_status()

        result = response.json()

        # completion endpoint의 경우
        if response_key == "response" and "response" in result:
            content = result["response"]
        # chat endpoint의 경우
        elif response_key == "message" and "message" in result:
            content = result["message"]["content"]
        else:
            print("❌ 응답에서 내용을 찾을 수 없습니다.")
            print(f"전체 응답: {result}")
            return None

        print("🤖 응답:")
        print(content)
        print("✅ 완료!")
        return content



if __name__ == '__main__':

    # Ollama API 클라이언트 초기화
    client = OllamaAPIClient()
    # 6. 추가 매개변수를 사용한 예제
    print("\n" + "=" * 60)
    client.generate_completion(
        "너를 소개해줘.",
        stream=False,
        options={
            "num_ctx": 30720,
            "temperature": 0.8,  # 창의성 증가
            "num_predict": -1,  # 토큰 수 제한, max_tokens
        }
    )
"""
    # 1. 모델 목록 조회
    models = client.list_models()
    print(models)

    # 2. 단순 텍스트 생성 (일반 방식)
    print("\n" + "=" * 60)
    client.generate_completion(
        "너를 소개해줘.",
        stream=False
    )

    # 3. 단순 텍스트 생성 (스트리밍 방식)
    print("\n" + "=" * 60)
    client.generate_completion(
        "Software를 개발할 때, 너에게 도움을 요청하는 방법을 알려줘?",
        stream=True
    )

    # 4. 대화 형식 (일반 방식)
    print("\n" + "=" * 60)
    messages = [
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user", "content": "오늘 날씨가 좋네. 산책하기 좋은 장소를 추천해줘."}
    ]
    client.chat_completion(messages, stream=False)

    # 5. 대화 형식 (스트리밍 방식)
    print("\n" + "=" * 60)
    messages = [
        {"role": "system", "content": "당신은 Python 전문가입니다."},
        {"role": "user", "content": "Python을 만든 사람은 누구인가요?"}
    ]
    client.chat_completion(messages, stream=True)
"""
