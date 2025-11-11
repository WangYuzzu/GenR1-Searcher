from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        question = data["question"]
        idx = data["idx"]

        messages_chat=[
            {"role": "system","content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".

During the thinking process, you have access to two external tools if necessary:

**Tool 1: Search System**
- Format: "<|begin_of_query|> search query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_query|>"
- Function: Retrieves relevant information from external knowledge bases
- Response format: "<|begin_of_documents|> ...search results... </|end_of_documents|>"

**Tool 2: Document Generation System**
- Format: "<|begin_of_generation|> generation query (only list keywords, such as "keyword_1 keyword_2 ...")</|end_of_generation|>"
- Function: A 7B parameter language model that generates documents tailored to your query
- Response format: "<|begin_of_documents|> ...generated document... </|end_of_documents|>"

**Note**: For multi-hop problems, it is recommended to decompose them into multiple simple sub-problems before using the tool. Each sub-problem should contain only a single query goal."""},
            {"role": "user", "content":question}
        ]

#         messages_chat = [
#             {"role": "system", "content": """You are a helpful assistant.
#         Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
#         The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
#         During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)</|end_of_query|>". **A query must involve only a single triple**.
#         Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... </|end_of_documents|>".
#         """},
#             {"role": "user", "content": question}
#         ]

        prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"

    else:
        print('用的base_prompt:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        base_prompt = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)</|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... </|end_of_documents|>".

User:{question}
Assistant: <think>"""

        question = data["question"]
        idx = data["idx"]
        prompt = base_prompt.format(question=question)

    return str(idx) + "<|idx_prompt_split|>" + prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)
        # print("len(self.prompts):",len(self.prompts))
        # print("self.prompts[0:5]:",self.prompts[0:5])
        # kill

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
