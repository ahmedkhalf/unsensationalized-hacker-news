import json
import subprocess as sp
from dataclasses import dataclass

import openai
import tiktoken

openai.api_key_path = ".key"


@dataclass
class Article:
    title: str
    content: str


class PostlightParser:
    @classmethod
    def parse(cls, link: str):
        cmd = ["postlight-parser", link, "--format=text"]
        parser = sp.Popen(cmd, stdout=sp.PIPE, text=True)
        stdout, _ = parser.communicate()
        if parser.returncode == 0:
            output = json.loads(stdout)
            return Article(
                title=output["title"].strip(), content=output["content"].strip()
            )
        return None


class PromptGen:
    TITLE_PROMPT = ""

    def __init__(self, max_tokens=2048 - 50, model="text-davinci-003"):
        self.max_tokens = max_tokens
        self.model = model

    @classmethod
    def _generate_title_prompt(cls, title: str, contents: str):
        if not cls.TITLE_PROMPT:
            with open("./title-prompt.md", "r") as f:
                cls.TITLE_PROMPT = f.read().strip()
        return cls.TITLE_PROMPT.format(title, contents)

    def generate_title_prompt(self, article: Article) -> str:
        enc = tiktoken.encoding_for_model(self.model)

        prompt_len = len(enc.encode(self._generate_title_prompt(article.title, "")))
        content = enc.encode(article.content)
        content_len = len(content)
        ommited = enc.encode("\n\n... content ommited ...\n\n")
        ommited_len = len(ommited)

        # another minus 50 for safety
        if prompt_len + content_len > self.max_tokens - 50:
            remove_len = content_len + prompt_len + ommited_len - self.max_tokens
            remove_i = content_len // 2 - remove_len // 2
            content = content[:remove_i] + ommited + content[remove_i + remove_len:]
            content = enc.decode(content)
            return self._generate_title_prompt(article.title, content)

        return self._generate_title_prompt(article.title, article.content)

    def infer_title(self, article: Article) -> str:
        prompt = self.generate_title_prompt(article)
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=0.15,
            max_tokens=50,
            stop="\n",
        )
        return response.choices[0].text.strip()


if __name__ == "__main__":
    article = PostlightParser.parse("https://openfolder.sh/heroku-anti-dx")
    promptGen = PromptGen()
    # if article:
    #     print(promptGen.infer_title(article))
