import os

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                model_name: str = "gpt-3.5-turbo-0125",
                model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                true_answer: str = None,
                question_asked: str = None,):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        # Force the model to use the one available in vLLM
        self.model_name = os.getenv("EVALUATOR_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        # Use dummy API key for vLLM
        api_key = os.getenv('NIAH_EVALUATOR_API_KEY', 'dummy-key')
        if (not api_key):
            api_key = "dummy-key"
            os.environ['NIAH_EVALUATOR_API_KEY'] = api_key
        
        self.api_key = api_key
        
        # Use custom base URL if provided
        base_url = os.getenv('OPENAI_EVAL_API_BASE', os.getenv('OPENAI_API_BASE'))
        
        # Initialize the evaluator with optional base_url
        self.evaluator = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=base_url,
            **self.model_kwargs
        )

    # def __init__(self,
    #              model_name: str = "gpt-3.5-turbo-0125",
    #              model_kwargs: dict = DEFAULT_MODEL_KWARGS,
    #              true_answer: str = None,
    #              question_asked: str = None,):
    #     """
    #     :param model_name: The name of the model.
    #     :param model_kwargs: Model configuration. Default is {temperature: 0}
    #     :param true_answer: The true answer to the question asked.
    #     :param question_asked: The question asked to the model.
    #     """

    #     if (not true_answer) or (not question_asked):
    #         raise ValueError("true_answer and question_asked must be supplied with init.")

    #     self.model_name = model_name
    #     self.model_kwargs = model_kwargs
    #     self.true_answer = true_answer
    #     self.question_asked = question_asked

    #     # Get API key or set a dummy key for vLLM
    #     api_key = os.getenv('NIAH_EVALUATOR_API_KEY', 'dummy-key')
    #     if not api_key:
    #         api_key = "dummy-key"
    #         os.environ['NIAH_EVALUATOR_API_KEY'] = api_key
        
    #     self.api_key = api_key
        
    #     # Get base_url for vLLM if set
    #     base_url = os.getenv('OPENAI_EVAL_API_BASE', os.getenv('OPENAI_API_BASE', None))
        
    #     # Initialize the evaluator with optional base_url
    #     if base_url:
    #         self.evaluator = ChatOpenAI(
    #             model=self.model_name,
    #             openai_api_key=self.api_key,
    #             openai_api_base=base_url,
    #             **self.model_kwargs
    #         )
    #     else:
    #         self.evaluator = ChatOpenAI(
    #             model=self.model_name,
    #             openai_api_key=self.api_key,
    #             **self.model_kwargs
    #         )

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])
