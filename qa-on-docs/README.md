This is a simple experiment to see if we can use langchain to answer questions about documentation.

It allows for taking a set of text files in a directory, then using embeddings and redis-based similarity search to narrow down
the context to feed to the LLM, and finally, by feeding the LLM both the context and the question, get an answer on
a completely arbitrary dataset.

### How to:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python run.py <dir>
```

You have two options for the `<dir>` argument:

"nothing" - that's basically no context

```
$ REBUILD_INDEX=true python run.py nothing
```

"eryndor" - a set of files about a fictional world

```
$ REBUILD_INDEX=true python run.py eryndor.
```
