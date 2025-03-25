import tiktoken

class TokenCounter:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count_tokens(text) for text in texts]

    def total_tokens(self, texts: list[str]) -> int:
        return sum(self.count_batch(texts))
