#!/usr/bin/env python3
"""
Script para limpar completamente o banco de dados vetorial.
Este script remove todas as perguntas e respostas armazenadas no Chroma DB.
"""
import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Carrega as variáveis de ambiente
load_dotenv()

# Configuração
VECTOR_DB_PATH = "./chroma_db"  # Caminho para o banco de dados Chroma

def clear_vector_database():
    """
    Limpa completamente o banco de dados vetorial de duas formas:
    1. Primeiro tenta usar o método interno do Chroma para deletar todos os documentos
    2. Se isso falhar, remove fisicamente o diretório e cria um novo
    """
    print("\n" + "=" * 60)
    print("LIMPEZA DO BANCO DE DADOS VETORIAL")
    print("=" * 60)
    
    # Método 1: Tentar usar a API do Chroma para remover todos os documentos
    try:
        print("Tentando limpar o banco de dados usando a API do Chroma...")
        embeddings = OpenAIEmbeddings()
        db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
        
        # Obter todos os IDs de documentos
        collection = db._collection
        ids = collection.get()['ids']
        
        if ids:
            print(f"Encontrados {len(ids)} documentos para excluir.")
            # Excluir todos os documentos
            collection.delete(ids)
            if hasattr(db, 'persist'):
                db.persist()
            print("Documentos excluídos com sucesso.")
        else:
            print("Nenhum documento encontrado no banco de dados.")
        
        print("Limpeza via API concluída com sucesso.")
        success_method = "API"
        
    except Exception as e:
        print(f"Erro ao limpar via API: {e}")
        print("Tentando o método alternativo...")
        success_method = None
    
    # Método 2: Remover fisicamente o diretório (fallback)
    if success_method is None:
        try:
            print("Removendo fisicamente o diretório do banco de dados...")
            
            if os.path.exists(VECTOR_DB_PATH):
                # Remove o diretório e todo seu conteúdo
                shutil.rmtree(VECTOR_DB_PATH)
                print(f"Diretório {VECTOR_DB_PATH} removido com sucesso.")
                
                # Cria um novo diretório vazio
                os.makedirs(VECTOR_DB_PATH, exist_ok=True)
                print(f"Novo diretório {VECTOR_DB_PATH} criado.")
                
                # Inicializa um novo banco de dados vazio
                embeddings = OpenAIEmbeddings()
                db = Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=embeddings
                )
                if hasattr(db, 'persist'):
                    db.persist()
                
                print("Novo banco de dados vazio inicializado.")
                success_method = "Físico"
            else:
                print(f"O diretório {VECTOR_DB_PATH} não existe. Criando um novo...")
                os.makedirs(VECTOR_DB_PATH, exist_ok=True)
                print("Diretório criado com sucesso.")
                success_method = "Físico"
                
        except Exception as e:
            print(f"Erro ao remover/recriar o diretório: {e}")
            success_method = None
    
    # Resultado final
    if success_method:
        print("\n" + "=" * 60)
        print(f"SUCESSO: Banco de dados limpo (Método: {success_method})")
        print("=" * 60)
        print("Todas as perguntas e respostas foram removidas.")
        print("O sistema agora começará com um banco de dados vazio.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FALHA: Não foi possível limpar o banco de dados")
        print("=" * 60)
        print("Tente manualmente remover o diretório:")
        print(f"rm -rf {VECTOR_DB_PATH}")
        print("=" * 60)

if __name__ == "__main__":
    confirmation = input("Isso irá EXCLUIR PERMANENTEMENTE todas as perguntas e respostas armazenadas. Continuar? (sim/não): ")
    if confirmation.lower() in ['sim', 's', 'yes', 'y']:
        clear_vector_database()
    else:
        print("Operação cancelada.")