# Exploring RAG

Repo for a coursework in Saint-Petersburg State University

# Introduction

Repo contains some implementations of RAG systems for a dedicated dataset provided by Sber Risks. 

You can find raw dataset in RTF (Rich Text Format) in  ```dataset/raw/*.rtf```.


# Evaluation techniques

TODO


# Notes

пайплайн из ноута залить на гитхаб и доработать
self-rag adaptive-rag
поресерчить м
в качестве n-gramm метрики взять готовые детерминированные bleu, roush...


## TODO 

- [x] Разобраться с датасетом сбера
- [ ] Запустить на нем простейший RAG
    - Сделать разбиение документов на чанки ограниченных размеров
    - Сделать простую индексацию через подсчет эмбеддингов + argmax(cosine_distance()). Посчитать качество top@k
    - Сделать простую генерацию поверх этой индексации (например через openrouter)
- [] create e2e quality measuring pipeline

- [] generate synthetic dataset with gpt