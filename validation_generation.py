import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from ast import literal_eval
from tqdm import tqdm

# from tokens import GIGACHAT_AUTHORIZATION_KEY
# from langchain.chat_models.gigachat import GigaChat


def compute_bleu(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=SmoothingFunction().method1)


def compute_rouge(reference, hypothesis):
    '''
    ROUGE-1: Measures the overlap of unigrams (individual words) between the generated text and the reference text.
    ROUGE-2: Measures the overlap of bigrams 
    ROUGE-L: Measures the longest common subsequence (LCS) between the generated and reference text. 
        It reflects the sequence similarity and accounts for in-order matches.
    r, p, f: recall, precision, and F1-score, respectively.
    '''
    rouge = Rouge()
    return rouge.get_scores(hypothesis, reference)[0]


# model = GigaChat(
#     credentials=GIGACHAT_AUTHORIZATION_KEY,
#     scope="GIGACHAT_API_PERS",
#     model=["GigaChat", "GigaChat-Pro"][0],
#     # Отключает проверку наличия сертификатов НУЦ Минцифры
#     verify_ssl_certs=False,
# )

def evaluate_with_llm(question, golden_answer, rag_answer, model):
    prompt = f"""
    Evaluate the quality of the answer provided by the model for given question, considering golden answer on this question.
    Question: {question}
    Golden Answer: {golden_answer}
    Model Answer: {rag_answer}
    Please provide a score between 1 (very poor) and 10 (excellent) based on relevance, correctness, and completeness.
    Please answer with dict with 2 key values, description - a string explaining score, score - integer value from 1 to 10."""
    response = model.invoke(prompt)
    answer = response.content
    answer = '\n'.join(answer.split('\n')[1:-1])  # answer looks like ```json \n{'description:'...} \n```
    answer = literal_eval(answer)
    return answer


def calculate_generation_metrics_v1(df: pd.DataFrame, model_generation=None):
    '''
    df: DataFrame with columns:
        question: string
        golden_answer: string
        rag_answer: string  
        retrieved_contexts: list[str] - list of retrieved contexts sorted from most relevant to least relevant
    '''

    for col in ['question', 'golden_answer', 'rag_answer','retrieved_contexts'][:2]:
        assert col in df.columns, f'Column {col} is missing in df'

    assert model_generation is not None, "model_generation arg is missing, object with function: def invoke(self, s:string='') -> string"
    
    # Metric: BLEU
    df['bleu'] = df.apply(lambda row: compute_bleu(row['golden_answer'], row['rag_answer']), axis=1)


    # Metric: Rouge
    df['rouge'] = df.apply(lambda x: compute_rouge(x['golden_answer'], x['rag_answer']), axis=1)  # each value is dict
    rouge_format = {'rouge-1': {'r': 0., 'p': 0., 'f': 0.},
                    'rouge-2': {'r': 0., 'p': 0., 'f': 0.},
                    'rouge-l': {'r': 0., 'p': 0., 'f': 0.}}
    for k1 in rouge_format:
        for k2 in rouge_format[k1]:
            df[k1 + '_' + k2] = df.apply(lambda x: x['rouge'][k1][k2], axis=1)

    df.drop(columns=['rouge'], inplace=True)


    # Metric: LLM score
    df['llm_score1_desc'] = ''
    df['llm_score1'] = ''
    for i, row in tqdm(df.iterrows(), total=len(df)):
        llm_metrics = evaluate_with_llm(row['question'], row['golden_answer'], row['rag_answer'], model_generation)
        df.loc[i, 'llm_score1_desc'] = llm_metrics['description']
        df.loc[i, 'llm_score1'] = llm_metrics['score']

    return df