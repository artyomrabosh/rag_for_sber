import numpy as np

from rank_bm25 import BM25Okapi

import nltk
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords

from langchain.schema import HumanMessage, SystemMessage

from langchain.prompts import PromptTemplate

import torch

nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")

def clean_text(text):
    text = WordPunctTokenizer().tokenize(text)
    text = [token.lower() for token in text if token.isalpha() and token not in russian_stopwords]
    return text


def fusion_retrieval_block(db, query, alpha=0.9, top_k=10):

    db_retrieved_scores = db.similarity_search_with_relevance_scores(query, len(db))

    clean_texts = [clean_text(doc[0].page_content) for doc in db_retrieved_scores]
    docs = [doc[0] for doc in db_retrieved_scores]
    BMdb = BM25Okapi(clean_texts)
    bm25_scores = BMdb.get_scores(clean_text(query))

    vector_scores = np.array([score for _, score in db_retrieved_scores])
    vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    if np.max(bm25_scores) - np.min(bm25_scores) == 0:
        bm25_scores = 0
    else:
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    combined_scores = alpha * vector_scores + (1-alpha) * bm25_scores

    sorted_indices = np.argsort(combined_scores)[::-1]

    return [docs[i] for i in sorted_indices[:top_k]], [db_retrieved_scores[i][1] for i in sorted_indices[:top_k]]



def llm_ranker(query, answers, model):

    MODEL_INSTRUCTION = 'Ты юрист в банковской сфере. Отвечай на вопросы на основе Положения Банка России \
"О требованиях к системе управления операционным риском в кредитной организации \
и банковской группе."'

    template = """Ответ должен содержать ровно одно число. Оцени по шкале от 1 до 10, насколько хорошо данный ответ отвечает на заданный вопрос.\
Вопрос: {query}\
Ответ: {answer}"""
    prompt = PromptTemplate(template=template, input_variables=['query', 'answer'])
    scores = []
    for answer in answers:
        PROMPT = prompt.format(query=query, answer=answer.page_content)

        messages = [
            SystemMessage(
                content=MODEL_INSTRUCTION
            ),
            HumanMessage(content=PROMPT)
        ]
        scores.append(float(model(messages)))

        return scores

def cross_encoder_ranker(query, answers, reranker_model):
    answers = [answer.page_content for answer in answers]
    rank_result = reranker_model.rank(query, answers)
    vals = [res['score'] for res in rank_result]
    return vals


def chat(model, prompt):
    MODEL_INSTRUCTION = 'Ты юрист в банковской сфере. Отвечай на вопросы на основе Положения Банка России \
"О требованиях к системе управления операционным риском в кредитной организации \
и банковской группе."'

    messages = [
        SystemMessage(
            content=MODEL_INSTRUCTION
        ),
        HumanMessage(content=prompt)
    ]

    res = model(messages)
    return res.content


class Retriever:
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, has_answer_th = 0.):
        self.database = db
        self.reranker = reranker
        self.strategy = strategy
        self.fusion_alpha = fusion_alpha
        self.k = k
        self.has_answer_th = has_answer_th

    def rerank_docs(self, query, docs):
        if self.reranker is None:
            return np.zeros(len(docs))
        return self.reranker(query, docs)

    def retrieve(self, query):
        query = self.get_advance_query(query)
        if self.strategy == 'mmr':
            docs = self.database.max_marginal_relevance_search(query, self.k)
            scores = self.rerank_docs(query, docs)

        else:
            docs, fusion_scores = fusion_retrieval_block(self.database, query, self.fusion_alpha, self.k)
            reranker_scores = self.rerank_docs(query, docs)
            scores = reranker_scores + fusion_scores

        score_doc = list(zip(scores, docs))
        score_doc.sort(key=lambda x: x[0])
        if score_doc[0][0] < self.has_answer_th:
            return None
        return [elem[1] for elem in score_doc]

    def get_advance_query(self, query): # оболочка для продвинутого класса
        return query


class EnrichAsAnswerRetriever(Retriever):
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, has_answer_th = 0., chat_model=None):
        super().__init__(db, reranker, strategy, fusion_alpha, k, has_answer_th)
        self.chat_model = chat_model

    def get_advance_query(self, query):
        PROMPT = 'Приведи пример ответа, который можно найти в Положении Банка России \
"О требованиях к системе управления операционным риском в кредитной организации \
и банковской группе."\n\
Вопрос: Расскажи про учет изменений в иностранном законодательстве при управлении оп риском в зарубежных дочерних кредитных организациях?\n\
Ответ: Кредитная организация должна учитывать требования национального законодательства иностранного государства при управлении операционным риском в дочерних организациях, включая порог регистрации событий. При этом показатели операционного риска приводятся в соответствие с требованиями национального законодательства, если оно противоречит требованиям Положения.\n\
Вопрос: Как кредитная организация должна учитывать потери от реализации событий операционного риска при расчете капитала, и какие требования предъявляются к ведению базы событий в этом контексте?\n\
Ответ: Кредитная организация должна ежемесячно определять величину валовых потерь от реализации событий операционного риска и использовать эти данные при выборе подхода к расчету объема капитала на покрытие таких потерь, выбирая между регуляторным и продвинутым подходами.\n\
Вопрос: '

        new_prompt = PROMPT + query + '\nОтвет: '
        return query + '\n' + chat(self.chat_model, new_prompt)


class EnrichAsQueryRetriever(Retriever):
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, has_answer_th = 0., chat_model=None, query_count=3):
        super().__init__(db, reranker, strategy, fusion_alpha, k, has_answer_th)
        self.chat_model = chat_model
        self.query_count = query_count

    def get_advance_query(self, query):
        PROMPT = f'Переформулируй вопрос {self.query_count} способами таким образом, чтобы ответ на них можно было найти в \
Положении Банка России \
"О требованиях к системе управления операционным риском в кредитной организации \
и банковской группе."\n \
Вопрос: '

        new_prompt = PROMPT + query
        return query + '\n' + chat(self.chat_model, new_prompt)


class EnrichAsCorrectionRetriever(Retriever):
    def __init__(self, db, reranker=None, strategy='mmr', fusion_alpha=1, k = 10, has_answer_th = 0., chat_model=None, query_count=3):
        super().__init__(db, reranker, strategy, fusion_alpha, k, has_answer_th)
        self.chat_model = chat_model
        self.query_count = query_count

    def get_advance_query(self, query):
        PROMPT = f'Ответь одним предложением.\nПереформулируй вопрос так, чтобы он стал более детальным и конкретным. Ответ на вопрос можно найти в \
Положении Банка России \
"О требованиях к системе управления операционным риском в кредитной организации \
и банковской группе."\n \
Вопрос: '

        new_prompt = PROMPT + query
        return query + '\n' + chat(self.chat_model, new_prompt)


class RAG:
    def __init__(self, retriever, model):
        self.chat_model = model
        self.retriever = retriever


    def get_answer(self, query, template):

        retrieved_documents = self.retriever.retrieve(query)
        if retrieved_documents is None:
            return 'Нет подходящей информации в данном документе'
        
        docs_page_content = [doc.page_content for doc in retrieved_documents]
        information = "\n\n".join(docs_page_content)
        prompt = PromptTemplate(template=template[0], input_variables=['information', 'query'])

        answer = chat(self.chat_model, prompt.format(information=information, query=query))
        return answer, retrieved_documents



class AdaptiveRAG(RAG):
    def __init__(self, retriever, model, classifier_model = None, max_cycle_iter=5, choise_strategy='greedy'):
        super().__init__(retriever, model)
        self.chat_model = model
        self.retriever = retriever
        self.classifier_model = classifier_model
        self.max_cycle_iter = max_cycle_iter
        self.choise_strategy = choise_strategy
        self.overlapping

    def query_complexity_classifier(self, query):
        if self.choise_strategy == 'greedy':
            return int(torch.argmax(self.classifier_model(query))) + 1

        if self.choise_strategy == 'max':
            return 3

        else:
            raise NotImplementedError

    def get_answer(self, query, template):

        query_complexity = self.query_complexity_classifier(query)

        if query_complexity == 1: # простой вопрос
            answer = chat(self.chat_model, query)
            return answer, None
        if query_complexity == 2: # средний вопрос
            retrieved_documents = self.retriever.retrieve(query)

            if retrieved_documents is None:
                return 'Нет подходящей информации в данном документе'

            docs_page_content = [doc.page_content for doc in retrieved_documents]
            information = "\n\n".join(docs_page_content)
            prompt = PromptTemplate(template=template[0], input_variables=['information', 'query'])

            answer = chat(self.chat_model, prompt.format(information=information, query=query))
            return answer, retrieved_documents
        if query_complexity == 3: # сложный вопрос
            num_docs = self.retriever.k
            for iter in self.max_cycle_iter:
                self.retriever.k = num_docs
                retrieved_documents = self.retriever.retrieve(query)
                docs_page_content = [doc.page_content for doc in retrieved_documents]
                information = "\n\n".join(docs_page_content)
                prompt = PromptTemplate(template=template[0], input_variables=['information', 'query'])

                answer = chat(self.chat_model, prompt.format(information=information, query=query))
                if answer is not None: # тут нужно определение того что модель ничего хорошего не ответила
                    return answer, retrieved_documents
                else:
                    num_docs *= 2


