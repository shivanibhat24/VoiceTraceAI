from setuptools import setup, find_packages

setup(
    name="voicetrace",
    version="1.0.0",
    description="Multimodal Voice Clone Attribution via Spectral CV and NLP Stylometry",
    author="Shivani Bhat",
    author_email="shivanibhat24@example.com",
    url="https://github.com/shivanibhat24/voicetrace",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0", "torchvision>=0.17.0", "torchaudio>=2.2.0",
        "librosa>=0.10.1", "soundfile>=0.12.1", "numpy>=1.26.0",
        "Pillow>=10.2.0", "opencv-python>=4.9.0", "timm>=0.9.16",
        "openai-whisper>=20231117", "transformers>=4.38.0",
        "sentence-transformers>=2.5.0", "spacy>=3.7.0", "nltk>=3.8.1",
        "yt-dlp>=2024.3.10", "feedparser>=6.0.11", "requests>=2.31.0",
        "gradio>=4.20.0", "fastapi>=0.110.0", "uvicorn>=0.28.0",
        "reportlab>=4.1.0", "networkx>=3.2.1",
        "pyyaml>=6.0.1", "tqdm>=4.66.2", "pandas>=2.2.0",
        "python-dotenv>=1.0.1", "loguru>=0.7.2", "scikit-learn>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "voicetrace=app.gradio_app:main",
            "voicetrace-api=app.api:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
