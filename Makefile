# VoiceTrace Makefile
# Author: Shivani Bhat | 20 March 2026
.PHONY: install scrape spectrograms train-cv build-db app api test clean

install:
	pip install -e .
	python -m spacy download en_core_web_sm
	python -m nltk.downloader punkt stopwords

scrape:
	python scraper/youtube_scraper.py --out data/raw --segment
	python scraper/podcast_scraper.py --out data/raw
	python scraper/synthetic_generator.py --input data/raw --out data/synthetic --tools coqui dummy

spectrograms:
	python cv_model/spectrogram.py --input data/raw --output data/spectrograms \
		--synthetic-manifest data/synthetic/synthetic_manifest.csv

train-cv:
	python cv_model/train.py --config configs/config.yaml

build-db:
	python nlp_model/speaker_db.py --audio data/raw --out data/speaker_db.pkl

app:
	python app/gradio_app.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

all: install scrape spectrograms train-cv build-db
