.PHONY: setup seed gate gate-data gate-train demo test api clean

setup:
	python -m spacy download en_core_web_sm
	python -c "from sentence_transformers import SentenceTransformer; \
	           SentenceTransformer('all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"

seed:
	python scripts/create_demo_docs.py

gate-data:
	PYTHONPATH=. python scripts/build_gate_data.py

gate-train:
	PYTHONPATH=. python scripts/train_gate.py

gate: gate-data gate-train

demo: seed gate
	@echo "Now run 'make api' and open http://localhost:8000"

api:
	PYTHONPATH=. uvicorn civsa.api:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=. pytest -v

clean:
	@read -p "This will delete indexes/ and documents/. Continue? [y/N] " ok; \
	 [ "$$ok" = "y" ] || [ "$$ok" = "Y" ] && rm -rf indexes/* documents/* || echo "aborted"

reindex:
	PYTHONPATH=. python scripts/reindex.py	 

eval:
	python -m scripts.eval

eval-sql:
	python -m scripts.eval --family sql

eval-http:
	python -m scripts.eval --http

eval-verbose:
	python -m scripts.eval --verbose	