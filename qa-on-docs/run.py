import os
import sys
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores.redis import Redis
from redis.client import Redis as RedisType
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()


def check_if_index_exists(client: RedisType, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except:
        return False
    return True


# Which models to use - this is using OpenAI now, but you can use any model
# I tried GPT4All/Llama, but it produced very bad embeddings and was too slow
embeddings = OpenAIEmbeddings()
llm = OpenAI()

# Expect that we get the index name as an argument
# The index name is the topic of the chatbot
if len(sys.argv) < 2:
    print("Usage: python run.py <directory>")
    sys.exit(1)

# get index_name from args
index_name = sys.argv[1].strip()

# check if folder with name "index_name" exists
if not os.path.isdir(index_name):
    print(f"Folder {index_name} does not exist")
    sys.exit(1)

# load all documents from folder
loader = DirectoryLoader(index_name, glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Allow to rebuild index via an env var - great for debugging
should_rebuild_index = os.environ.get(
    "REBUILD_INDEX", "false").lower() == "true"

redis_url = os.environ['REDIS_URL']
redis = RedisType.from_url(redis_url)

if should_rebuild_index == False and check_if_index_exists(redis, index_name):
    print("Index exists, loading from Redis")
    rds = Redis.from_existing_index(
        embeddings, redis_url=redis_url, index_name=index_name)
else:
    redis.flushall()
    print("Rebuilding index")
    # index does not exist, create it
    # This will take the docs, produce embeddings and store them in Redis
    rds = Redis.from_documents(
        docs, embeddings, redis_url=redis_url,  index_name=index_name)


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                 retriever=rds.as_retriever(search_kwargs={"k": 1}))

print("Ready to answer questions!")
print("> ", end="", flush=True)

# read one line of input from stdin until ctrl+c
for line in sys.stdin:
    # add a ' > 'prompt to the input
    # flush the output to make sure the prompt is printed
    sys.stdout.flush()
    question = line.strip()
    result = qa.run(question)
    stripped = result.strip()
    print("<", stripped)
    print("> ", end="", flush=True)
