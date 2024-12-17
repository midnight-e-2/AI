from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from transformers import Embedding
from langchain_community.vectorstores import Chroma

load_dotenv()

class Dinosaur_Model():
    def __init__(self, dinosaur_name):
        self.model = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
        # RAG 설정: CSV 로더, 텍스트 분할기, 임베딩, 벡터 저장소 설정
        loader = CSVLoader(file_path='data.csv', encoding='utf8')
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        texts = text_splitter.split_documents(document)

        embedding = OpenAIEmbeddings()
        persist_directory = 'dino_db'
        self.vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=persist_directory
        )
        self.vectordb.persist()
        self.retriever = self.vectordb.as_retriever(search_kwargs={'k': 3})
        self.dinosaur_prompt = '''
            너는 초등학생 아이에게 공룡에 대해 알려줄거야.
            너는 {0} 공룡의 입장이 되어 설명할거야.
            공룡이 할 법한 말이 아니면 답변하지 마.
            답변을 조금 짧게 해줘.
            '''.format(dinosaur_name)
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f'{self.dinosaur_prompt}'),
            HumanMessagePromptTemplate.from_template('{history}'),
            HumanMessagePromptTemplate.from_template('{input}')
        ])
        self.memory = ConversationBufferMemory()
        
        self.chain = ConversationChain(
            llm = self.model,
            prompt = self.chat_prompt,
            memory = self.memory
        )
        
    def get_model(self):
        return self.chain
    
    def exec(self, query):
        # 벡터 DB에서 관련 문서 검색
        relevant_docs = self.retriever.get_relevant_documents(query)
        
        # 검색된 문서 내용을 바탕으로 프롬프트에 정보를 추가
        search_results = "\n".join([doc.page_content for doc in relevant_docs])
        enhanced_query = f"{query}\n\n[참고 자료]\n{search_results}"
        
        # LLM에 전달하여 답변 생성
        response = self.chain.run(input=enhanced_query)
        return response


if __name__ =="__main__":
    # 유저 1이 질문함
    user1_t_model = Dinosaur_Model('Tyrannosaurus')
    response = user1_t_model.handle_query('육식 공룡에 대해 알려줘. 난 토마토 좋아하는데 넌 뭘 좋아해')
    print(response)
    response = user1_t_model.handle_query('내가 뭘 좋아한댔지?')
    print(response)
    
    # 유저 2가 질문함
    user2_t_model = Dinosaur_Model('Tyrannosaurus')
    response = user2_t_model.handle_query('초식 공룡에 대해 알려줘')
    print(response)

    