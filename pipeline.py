import pandas as pd
import os
import json
from typing import List
import asyncio
import instructor

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, MistralEmbeddingService

def get_documents():
    documents = pd.read_csv("dataset/clean/risk1.csv")
    documents[documents.isna()] = ""
    documents["full_text"] = (documents['Header_1'] + " "
                              + documents['Header_2'] +" "
                              + documents['Header_3'] +" "
                              + documents['content'])
    return documents

def write_strings_to_files(strings, folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Iterate through the list of strings
    for index, string in enumerate(strings):
        # Create a file name for each string
        file_name = f"file_{index + 1}.txt"
        file_path = os.path.join(folder_path, file_name)

        # Write the string to the file
        with open(file_path, 'w') as file:
            file.write(string)


async def _save_graphml(state_manager, output_path) -> None:
    await state_manager.query_start()
    try:
        await state_manager.save_graphml(output_path)
    finally:
        await state_manager.query_done()

async def main():
    folder_path = "dataset/clean/risk1/"
    docs = get_documents()

    write_strings_to_files(list(docs.full_text), folder_path)
    DOMAIN = "Юридические документы Центрального Банка России"
    QUERIES: List[str] = ["В какие сроки головная кредитная организация банковской группы должна представить в Банк России отчеты о размере оп риска?", "Как учитываются корректирующие события после отчетной даты при перерасчете размера опер риска?"]
    ENTITY_TYPES: List[str] = ["Документ", "Организация", "Формула", ]

    working_dir = "./clean/risk1"
    
    with open('api_tokens.json', 'r') as file:
        api_tokens = json.load(file)

    embedding_service = MistralEmbeddingService(
        api_key=api_tokens["mistral_api_key"],
    )
    
    api_key = api_tokens["openai_api_key"]
    url_or = "https://openrouter.ai/api/v1"

    grag = GraphRAG(
        working_dir=working_dir,
        domain=DOMAIN,
        example_queries="\n".join(QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=OpenAILLMService(
                model="meta-llama/llama-3.1-70b-instruct:free", base_url=url_or, api_key=api_key, mode=instructor.Mode.JSON
            ),
            embedding_service=embedding_service
        ),
    )

    await grag.async_insert(list(docs.full_text))


    await _save_graphml(grag.state_manager, "graph.graphml")


if __name__ == '__main__':
    asyncio.run(main())



