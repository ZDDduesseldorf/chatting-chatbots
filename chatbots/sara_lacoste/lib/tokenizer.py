import tensorflow_datasets as tfds


class TransformerTokenizer:
    def __init__(self, tokenizer):
        self.start_token: int = tokenizer.vocab_size
        self.end_token: int = tokenizer.vocab_size + 1
        self.vocab_size: int = tokenizer.vocab_size + 2
        self.corpus_vocab_size: int = tokenizer.vocab_size
        self._tokenizer = tokenizer

    def build_from_corpus(corpus: list[str], target_vocab_size: int):
        return TransformerTokenizer(
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                corpus,
                target_vocab_size=target_vocab_size))

    def load_from_file(path: str):
        return TransformerTokenizer(
            tfds.deprecated.text.SubwordTextEncoder.load_from_file(path))

    def save_to_file(self, path: str):
        self._tokenizer.save_to_file(path)

    def encode(self, sentence: str) -> list[int]:
        return self._tokenizer.encode(sentence)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
