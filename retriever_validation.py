from scipy.spatial.distance import cosine
import numpy as np

def metric_docs_docs(retrieved_docs, true_docs, embedder):
    retrieved_embs = embedder.embed_docs_pc(retrieved_docs)
    true_embs = embedder.embed_documents(true_docs)

    res_1 = 0
    for true_emb in true_embs:
        min_dist = np.inf
        for doc_emb in retrieved_embs:
            min_dist = min(min_dist, cosine(true_emb, doc_emb))
        res_1 += min_dist

    res_2 = 0
    for doc_emb in retrieved_embs:
        max_dist = -np.inf
        for true_emb in true_embs:
            max_dist = max(max_dist, cosine(true_emb, doc_emb))
        res_2 += max_dist


    return 1 - res_1 / len(retrieved_docs), 1 - res_2 / len(retrieved_docs)


def metric_docs_answer(retrieved_docs, answer, embedder):
    res = 0
    retrieved_embs = embedder.embed_docs_pc(retrieved_docs)
    answer_emb = embedder.embed_query(answer)
    min_dist = np.inf
    max_dist = -np.inf

    for doc_emb in retrieved_embs:
        dist = cosine(answer_emb, doc_emb)
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)
        res += dist

    return 1 - res / len(retrieved_docs), 1-min_dist, 1-max_dist


def NotIsNaN(obj):
    return obj == obj


def count_metrics(retriever, top_k, queries, embedder):
    retr_k = retriever.k
    retriever.k = top_k


    doc_doc_metrics_prec = []
    doc_doc_metrics_rec = []

    doc_answer_metrics_mean = []
    doc_answer_metrics_max = []
    doc_answer_metrics_min = []

    confident_metrics = []

    for row in queries.iterrows():
        row = row[1].loc()
        query = row['Вопрос']
        answer = row['Ответ']
        answer_docs = []

        if NotIsNaN(row['Пункт1']): answer_docs.append(row['Пункт1'])
        if NotIsNaN(row['Пункт2']): answer_docs.append(row['Пункт2'])
        if NotIsNaN(row['Пункт3']): answer_docs.append(row['Пункт3'])

        retrieved_docs = retriever.retrieve(query)
        doc_doc_res = metric_docs_docs(retrieved_docs, answer_docs, embedder)

        doc_doc_metrics_prec.append(doc_doc_res[0])
        doc_doc_metrics_rec.append(doc_doc_res[1])

        doc_ans_res = metric_docs_answer(retrieved_docs, answer, embedder)

        doc_answer_metrics_mean.append(doc_ans_res[0])
        doc_answer_metrics_min.append(doc_ans_res[1])
        doc_answer_metrics_max.append(doc_ans_res[2])

        confident_metrics.append(1 - cosine(embedder.embed_query(retrieved_docs[0].page_content), embedder.embed_query(query)))

    res = {
        'doc_doc_metric_prec =': np.mean(doc_doc_metrics_prec),
        'doc_doc_metric_rec =': np.mean(doc_doc_metrics_rec),

        'doc_answer_metric_mean =': np.mean(doc_answer_metrics_mean),
        'doc_answer_metric_min =': np.mean(doc_answer_metrics_min),
        'doc_answer_metric_max =': np.mean(doc_answer_metrics_max),

        'confident_metric =': np.mean(confident_metrics),
    }

    retriever.k = retr_k
    
    return res
