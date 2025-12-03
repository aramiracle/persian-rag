from backend.services.embedding_service import get_embedding_service
from backend.services.llm_service import get_llm_service
from backend.services.milvus_client import get_milvus_client
from backend.services.elasticsearch_service import get_elasticsearch_service
from backend.services.rag_service import get_rag_service
from backend.services.upload_service import get_upload_service

def get_rag_service_dependency():
    return get_rag_service(
        milvus_client=get_milvus_client(),
        es_service=get_elasticsearch_service(),
        embedding_service=get_embedding_service(),
        llm_service=get_llm_service(),
    )

def get_upload_service_dependency():
    return get_upload_service(
        milvus_client=get_milvus_client(),
        es_service=get_elasticsearch_service(),
        embedding_service=get_embedding_service(),
    )
